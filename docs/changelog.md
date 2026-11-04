# 版本更新

按日期倒序，README 只列最近 5 条。

- 2026-11-04：**Turing（sm_75）这一档的 decode 提速**：decode 从 0.5-0.9 tok/s 到 **2.9-3.0 tok/s**（只算 decode，不含
  prefill；128 token 的流式请求实测 2.85/3.00/2.92/2.98 tok/s），同时修掉一个**静默算错**的地方。落地的东西：
  1. **TP 切分算错了**：GPU 池算的是整份专家（中间维 2304），而 SGLang 会把两个 rank 的 MoE 输出相加，所以每条路由
     被算了 2 遍。证据是同一段提示的总 logprob：池子这条路 **-188.6**，纯 CPU 引擎 **-168.1**。现在池子只算本 rank
     的那一片（gate/up 的行、down 的列，跟引擎的 `gsq_split` 一样）：池子这条路 **-169.7**（跟引擎的 BF16 只差 FP16
     的舍入），复制测试仍 100% 命中。`gsq_gpu.py` 的四个内核多了一个 `IOFF`/`IP`（本 rank 的中间维偏移与宽度），
     down 的边界块整块解码、按列掩掉（掩掉的列在中间维里是 0，点积不受影响）；
  2. **上传不再把 decode 流排空**：以前每个不在显存里的专家分三段拷，拷一次 `cudaStreamSynchronize` 一次，一层
     十六次，每次把整条 decode 流水线排空。现在拷贝走自己的流（decode 流用 event 等它，拷贝流等 decode 读完
     arena 再覆写），pinned 的暂存缓冲也做成环形。实测这一项就是 0.5 → 1.65 tok/s；
  3. **只传本 rank 的那一片字节**：gate/up 的行是连续的，所以显存里的槽只放本 rank 的行（5.2 MB 而不是 7.7 MB），
     PCIe 上的字节少三分之一，显存里能多放一倍的槽（26 个/层）。实测 1.65 → 2.2 tok/s；
  4. **自适应驻留真的在跟着流量走**：交换（swap）以前只能从引擎的 RAM 槽读字节，本机每层只有 384 个专家里的 115 个
     在内存里，于是 87% 的交换被跳过（`swaps_no_ram`）。现在专家不在内存里时从 checkpoint 自己的文件映射读，命中
     率从 8% 涨到 33%（每层路由量前 26 个专家占约三分之一）；
  5. **内存里的冷池跟着 GPU 的流量走**：上传以前是从 checkpoint 的文件里读专家的字节（引擎的 RAM 池里放的是 profile
     排出来的专家，引擎自己一条路由都没走到，因为路由全被 GPU 接管了），于是每次调用都从盘上读。现在池子在上传前
     让冷池把这次路由到的专家读进引擎的 RAM（跟引擎自己算之前做的是同一件事：不在 RAM 里的从 store 读进来，最冷的
     那个让位），RAM 里于是就是 GPU 真正要的专家，上传从已 pin 的槽拷（满速 PCIe），**稳态下 checkpoint 的文件一次都
     不读了**（上传里来自文件的字节：每层每 192 次调用 748 MiB -> 0）。这一项把 decode 从 2.2 抬到 2.5-2.9 tok/s；
  6. **内存里的冷池只存本 rank 的那一片**：RAM 池原来存整份专家（7.7 MB），而每个 rank 只需要自己那一片（5.2 MB：
     gate/up 的行、down 整个），于是同样的内存只放得下每层 115 个专家，而流量碰到的是 116-125 个，LRU 在颠簸、盘上
     每次调用都要读。现在槽里放的是本 rank 的那一片（引擎读槽里的行、down 的列，跟 GPU 读的一样），同样的内存放得下
     **172** 个，工作集装得下，盘只读一次（上传里来自文件的字节 0）。引擎自己的数字一点没变：CPU 引擎那条路用切片的槽
     算出来的 logprob 一模一样（-168.0674）。这一步把 decode 从 2.5-2.9 抬到 2.9-3.0 tok/s，内存也省了 29 GB；
  7. **冷池的预取**：路由最多、还不在内存里的专家现在由引擎的冷池在后台读（`xtier_gsq_cold_prefetch`），盘的延迟
     在 decode 跑的时候付掉，而不是等某次调用要用它的时候（一层调用的读盘 2.2 ms -> 1.75 ms）；
  8. **prefill 的 CPU 那半没了**：`XTIER_GSQ_PREFILL_UPLOAD=128`（暂存 arena 的槽数），批次路由到的专家全部由 GPU
     算，CPU 引擎每层只剩 1-20 ms（原来 200-660 ms）；prefill 的 tile 也调小了（`XTIER_GSQ_PREFILL_TILE_M=8`：32 个
     路由 × 16 行的 tile 寄存器放不下，decode 的内核已经用满 255 个）；IQ2_XXS 的解码也改成三维（行、八个值的一组、
     组里的值）：一组的码和比例对它的八个值是一样的，于是按组读一次而不是每个值读一次（每个值六次读变成一个半，kernel
     的时间都花在读上），**首 token 的时间从 47 s 降到 17 s**（decode 不变：它卡在幅度值的 gather 上，不是码上）；
  9. **15 tok/s 在这张卡上放不下**：decode 的 67 ms/token 预算，光是 GPU 上把路由到的专家的 IQ 块解开就要
     **85-130 ms/token**（1.4 G 个值，卡上 44 G 值/s），稠密权重本身又是 12.1 GB/卡（27 ms/token）。能压的是周围的
     等待，实测压到了 2.5-2.9 tok/s，剩下的时间是 PCIe 上那些显存里放不下的专家的字节（现在走内存，不再走盘）和每层
     几十次 kernel 发射的 CPU 开销（GPU 在两次发射之间是闲着的，而 CUDA graph 在这里用不了：上传的地址每次调用都
     变）。详见[文档](sm75.md)。
- 2026-11-03：**Turing（sm_75）这一档做成了从 main 出发的分支** `feat/sm75-turing`：把上游针对 30/40 系（sm_86/sm_89）
  的做法整套搬过来（分支 = main 的 `4b1e766` ＋ Turing 的那套活），再补上 RTX 30/40 不需要的一层：**这里没有 BF16
  张量核**。落地的东西：
  1. **dtype 层**：模型跑 **FP16**（卡上张量核吃的就是 FP16：实测 bf16 的 `mm` 6.2 TFLOP/s 对 fp16 的 106，Triton 的
     BF16 `tl.dot` 更慢），超连接的残差流留在 **FP32**（它 40 层涨到 284672，超过 FP16 的 65504；`dtypes.py` 的
     `_post_fp32`/`_combine_fp16`/`_norm_fp32`），CPU 引擎的 BF16 契约不变。上游那些按 BF16 模型写的内核改成跟着
     激活的 dtype 走（`wo_a`、`w8a16`、`engram_gather`、Engram 行缓冲、FP8 KV 缓存、Qwen 的 PLE 行与打包 KV、
     MiniMax 的 FP4 KV）：`arch.dtype()` 按卡取（sm_80 以下 FP16），内核用 `DT` 这个 constexpr；
  2. **合并处的一个断点**：`arch.e4m3_f32` 的两种调用风格（上游 1 个参数、Turing 的内核 2 个）合不拢，Triton 报
     `TypeError: too many positional arguments`，缓存的 gate/up 内核编不过；现在 `xtier/arch.py` 是唯一入口，硬件/软件
     转换按调用传（`HW`），两种写法都能用；
  3. 检查又找出三处合并伤，都修了：`_reduce` 的参数个数（上游的内核是 6 个）、`hybrid_prefill` 丢了上游的
     `finally: worker.join()`（GPU 那半失败时 CPU 那半还活着）、GPU prefill 的权重反量化把 E4M3 转换留在硬件默认
     （sm_89 以下编不过）；第二张卡当热专家池的那条路（`cache/second.py`）也补上了编解码标志；
  4. **开关**：mHC pre-mix 的 TileLang split-K 要 sm_80（下面关掉，TileLang 那一整套本来就在下面关着）；FP8/FP4 的
     功能按等级关闭（`supported()`）；
  5. **卡上的检查**：`sm75_gpu_check`（缓存的 gate/up 与 GPU prefill 对 FP32 参考，FP16 下相对 L2 2.1e-4/3.0e-4）、
     `fp4_decode_check`、`fp8_decode_check`、`fp8_encode_check`（940 万个值与 torch 的 e4m3fn 逐位相同）、
     `ds_fp8_soft_check`、`ds_sparse_mla_check`、`sparse_mla_check`、`ds_w8a16_check`、`deepseek_fp8_kv_check`、
     `qsa_decode_check`、`qsa_fp8_check`、`glm_indexer_torch_check`、`arch_check` 都通过；需要别的机器上的
     safetensors 检查点或要 sm_89/sm_90 的那几条在本机改成 SKIP（原来直接失败）；跟 SGLang 自己 FP8 内核对不上的那几条
     （`fp8_encode_check`、`ds_fp8_soft_check`、`qsa_fp8_check`）改成跟卡上有的参考比（torch 的 e4m3fn、池子自己的字节、
     SGLang 的 torch 路径）；
  6. **两张卡**：TP=2（每张卡一半的稠密权重、一半的 KV、各自的 CPU 引擎，`tools/cpu_budget.py --tp 2` 分线程），
     `XTIER_COLD_FRACTION=0.8`（两个 rank 的冷池各留 77/384 个专家在内存里，0.3 的话要 195 GB，cgroup 会卡住加载）。
     实测：两张卡都用满（各 20.9/22.5 GB），加载约 3 分钟（权重 167 s），一次请求（81 token 提示、64 token 输出）
     冷 198.7 s、热 187.1 s，答案正确（prefill 0.4-1.35 tok/s、decode 0.24-0.44 tok/s，SGLang 的计数）。
  详见[文档](sm75.md)。
- 2026-10-30：Turing（sm_75）上的 DeepSeek V4.1 Flash 跑通了：一次请求有答案（"2+2 equals 4."），40 层没有 NaN。
  实测速度：prefill 约 3.7 tok/s、decode 约 0.8 tok/s（SGLang 自己的计数），瓶颈是 BF16 的点积由软件模拟、MoE 在
  CPU 引擎上、注意力走 PyTorch，不是显存。修了四处：
  1. GGUF 的名字对照漏了共享专家的三个张量（gate/up/down），共享专家的权重一直没初始化：隐藏状态变 NaN，引擎拒绝
     路由权重（`xtier/backend/gguf_names.py`）；
  2. 压缩 KV 的融合 RoPE+FP4 假量化要转 NVFP4 的 e4m3，Triton 只在 sm_89 以上能编，改走 SGLang 自己的 torch 路径
     （`xtier/packs/deepseek_v4/fp4_rope.py`）；
  3. 索引器与 KV 压缩器的权重固定是 BF16，激活按它们的 dtype 走；BF16→FP32 的 GEMM 要 sm_80，改为上转型
     （`xtier/packs/deepseek_v4/dtypes.py`）；
  4. 模型的 dtype 改回 BF16：超连接的残差流在 FP16 下到第 23 层就撑爆 65504（上游也是 BF16）。
  端到端三层都在日志里：GPU 热缓存命中 7% 的路由（8 个槽对 384 个专家），其余由 CPU 引擎算（每层调用 35 ms），
  内存与 GPU 槽里都没有的专家由冷池从傲腾盘读（每步约 1 个）；另外把分块 prefill 的块从 1024 改成 256（便携注意力
  一次 gather 1.75 GiB，卡上只剩 1.25 GiB），GSQ 冷池也补上了统计日志（`XTIER_COLD_LOG`）。
  详见[文档](sm75.md)。
- 2026-11-02：Turing（sm_75）照搬别人提交的 **Ampere（RTX 30/40）适配**的做法：**不改检查点的存储格式，用软件适配**——
  权重反量化到显卡能算的格式再算。落地了三件事：
  1. E2M1/E4M3 的软硬件转换都齐了（`xtier/backend/fp4.py`）：E2M1 解码、E4M3 解码与编码，位级等于硬件指令
     （`tests/fp4_decode_check.py`、`tests/fp8_decode_check.py`、`tests/fp8_encode_check.py`），
     `XTIER_FP4_DECODE` / `XTIER_FP8_DECODE` 选路径，可在同一张卡上对拍；
  2. 32×32 分块 FP8 的投影在 sm_89 以下也能跑（`xtier/packs/deepseek_v4/w8a16.py`）：保持 SGLang 自己的布局
     （不重排 Marlin、不动检查点），权重按位反量化进复用的 BF16 buffer 再走 cuBLAS；
  3. 记下一条量到的数：超连接的残差流在 40 层里涨到 **284672**（FP16 的上限是 65504），所以模型的 dtype 必须是
     BF16——卡上没有 BF16 张量核，这部分是速度的主要代价。单次实测（prompt 126 + decode 64）：全程 124 s，
     decode 0.5 tok/s（SGLang 计数 0.40→0.76），冷 prefill 约 1 tok/s、热 prefill 3.7 tok/s。
- 2026-10-29（夜）：**GGUF 分片**：SGLang 的 GGUF loader 读的是**一个**文件（`gguf` 包的 reader 没有分片支持：
  第一个分片里只有 42 个张量，整个检查点 1008 个），DeepSeek V4.1 的 Maya 包是四个分片。XTier 现在把分片当一个
  文件读（`xtier/backend/gguf_split.py`），给出该模型的张量名对照（`gguf_names.py`）与权重的反量化
  （`gguf_weights.py`：模型的注意力要反量化的权重，路由专家留在文件里给引擎的 IQ 解码）。
- 2026-10-29（晚）：GSQ 路径（GGUF 的 IQ 专家）补上本项目的两个主功能：**热专家缓存**（GPU 用 Triton 解 IQ 块，
  热专家的字节按检查点的原样进显存，常驻集跟着路由换人；实测同一批路由 GPU 22.9 ms 对引擎 90.5 ms，相对 L2 3.8e-04）
  与**冷专家卸载到傲腾盘**（`XTIER_COLD=1`：热专家进内存槽，其余留在检查点自己的分片里，O_DIRECT 读，不经页缓存；
  本机实测傲腾 2.4–3.5 GB/s，30% 卸载时 decode 每步缺 0.4 个专家、每步读 3.1 MB，步长与不卸载时相同）。
  详见[文档](sm75.md)。
- 2026-10-29：Turing（sm_75）适配，分支 `feat/sm75`：GPU 内核按显卡的算力等级取 dtype 与实现（Turing 没有 BF16
  张量核，模型以 FP16 跑，下投影用 16 位点积；没有 FP4 转换指令，热专家缓存用软件解码 E2M1），加速包里的 FP8/TMA/FP4
  功能按等级关闭，tile 按 64 KiB 共享内存缩小；新增 2× RTX 2080 Ti 的配置。详见[文档](sm75.md)。
- 2026-11-01（v0.1.1）：修了 DeepSeek 在 RTX 30 / 40 系上第一个请求就崩（`Unsupported architecture for sparse decode fwd`）。SGLang 10-23 把 V4 后端改成
  经 `flash_mla_interface` 调 FlashMLA，XTier 10-30 跟进 6162b07 后，`packs/deepseek_v4` 的钩子还挂在旧入口上；旧入口还在，所以启动检查查不出
  （iounstone 发现，[#10](https://github.com/a775828b-dot/xtier/pull/10)）。修复后在 RTX PRO 5000 上模拟 SM86 跑整模型：照抄 0 错，NLL 1.19735（升级前 1.19867）。真卡还要等他确认。
  发布前清单加了模拟 SM86 这一步。
- 2026-11-01（v0.1.0）：全面代码审计后的修复。稳定性：DeepSeek 的 CUTLASS 内核在加载权重前编译（不再在第一个请求里现编、被内存守护杀掉）；换槽序号回绕；split
  prefill、混合 prefill、冷专家写盘的线程报错不再被吞；专家没加载全、钩子没生效、调度入口改名都在启动时报错；未知的 `XTIER_*` 启动时警告；测试 FAIL 一律非零退出，加
  `tools/repo_check.py` 和 CI。查明同一提示词重算 logprob 摆动来自 top-k 输出顺序，不是 bug；几项换槽参数和提示词预热 A/B 没有收益，默认不变。详见[文档](reports/2026-11-01-audit-fixes.md)。
  冷专家：池子换到 4 块 PCIe 5.0 盘，DeepSeek 卸载 10 / 20 / 30 / 40 / 50% 时 decode 每步 31.6 / 34.4 / 37.9 / 44.5 / 53.5 ms（不卸载 30.0）；`tools/cold_punch.py` 让大模型的专家只存一份
  （逐字节核对后在模型文件里打洞，DeepSeek 的副本 290 → 21 GiB，`--restore` 写回）。详见[文档](reports/2026-11-01-cold-pcie5.md)。
  文档：新增[文档目录](README.md)、[快速开始](quickstart.md)（从空机器到第一次请求，新环境实跑过）、[支持情况](support.md)（实际验证到什么程度）、[常见问题](faq.md)、[发布前检查清单](release-checklist.md)和英文版 [README](README.en.md)、[快速开始](quickstart.en.md)；
  `constraints.txt` 固定依赖版本，`tools/check_sglang.py --strict`；测速表的“官方 main”改叫“基线”，它带着 XTier 的混合推理，不是原版 SGLang。
- 2026-10-31：第二张卡可以不放模型、纯当热专家池（可选，`XTIER_GPU2_SLOTS=N|auto`，默认关；`TP=1`），给显存或速度不一样的两张卡用：
  主卡不变，信箱调度线程把主卡没命中、第二张卡有的专家交给它，结果经内存加回 CPU 输出，每次调用分给它多少专家自动调整。双 RTX 4090
  模拟 24 + 16 GiB：Qwen3.8 decode 57.3 / 55.9 tok/s（32K / 120K），同样配置的 TP 51.1 / 49.4，单卡 41.0 / 43.6；第二张卡约慢 1.2 倍时
  54.5 / 54.2。同样的专家放第二张卡和放主卡，单层结果逐位相同（Qwen3.8、GLM-5.3、DeepSeek V4.1）。真实异构卡还没测过。功能关时
  三个模型的照抄、PPL、长上下文与上次一致，dsh 连续 15 分钟无错。详见[文档](reports/2026-10-31-second-gpu-pool.md)。
- 2026-10-30（下午）：跟进官方 SGLang main `6162b07199`（10-22 的 `8cd3c494` 之后 592 个提交）。XTier 跟着改了 Engram 行放
  NVMe、MoE 方法的 top-k、Qwen PLE 分段 prefill；DeepSeek 的 prompt 改由上游用 deepseek-recipe 编码，删掉 `recipe_tokens`。测试过的
  提交写进安装步骤和 `xtier.SGLANG_COMMIT`，装的版本不同时启动会警告，`tools/check_sglang.py` 核对钩子目标与 import。三个模型照抄、
  PPL 与旧版一致，512K 不 OOM，dsh 连续 15 分钟无错。另外：快速加载支持 MoE 张量并行（双 RTX 4090 实测通过：分片与普通加载逐字节相同）；英伟达官方
  GLM NVFP4 检查点（NextN 层是 BF16）开 MTP 不再崩溃，草稿层的专家加载时转成 NVFP4 交给 CPU 引擎；启动台在 chroot 等没有用户
  systemd 的环境里退回 setsid。详见[文档](reports/2026-10-30-sglang-6162b07.md)。
- 2026-10-30：双卡张量并行（`TP=2`）实测通过：双 RTX 3090 上 Qwen3.8 decode +42–48%、120K 首字 −35%，照抄 0 错；修了
  `~/.cache` 是软链接时 SGLang 的 JIT 内核每次启动都重编（30 系单卡启动省约 5 分钟）。详见[文档](reports/2026-10-30-tp.md)。
- 2026-10-29（晚）：支持 RTX 30 系（SM86）：没有 FP8 硬件，FP8 一律读时软件解码（与硬件逐位相同），DeepSeek 的 FP8
  投影只量化权重；RTX 3090 实测 Qwen3.8 整模型照抄 0 错、120K 上下文。详见[文档](reports/2026-10-29-rtx30.md)。
- 2026-10-29：支持 RTX 40 系（SM89）：FP4 读时软件解码（与硬件逐位相同，不增加精度损失），三个模型的加速包补上 SM90
  以下的内核；RTX 4090 实测 Qwen3.8 整模型照抄 0 错、120K 上下文。详见[文档](reports/2026-10-29-rtx40.md)。
- 2026-10-28：decode 时可把一部分未命中的专家交给 GPU 代算（可选，`XTIER_DECODE_MISS_GPU`，默认关；启动台三张卡可选），
  给 CPU 比 PCIe 慢的机器用，结果与命中的专家相同。详见[文档](reports/2026-10-28-decode-miss-gpu.md)。
- 2026-10-27（下午）：快速加载（可选，`XTIER_FAST_LOAD=1`，默认关）：专家直接从模型文件或冷存储读进 CPU 引擎，仍是全部装好
  才报在线；DeepSeek 启动 472 → 186 s，卸载 30% 409 → 124 s。详见[文档](reports/2026-10-27-fast-load.md)。
- 2026-10-27：启动台新卡片「DeepSeek V4.1 Flash 1017 · 最小化」：32 GB 显存 + 128 GB 内存（模拟 RTX 5090），70% 的专家放硬盘，
  prefill 约 1,160、decode 约 30 tok/s。详见[文档](reports/2026-10-27-ds41-minimal.md)。
- 2026-10-26（晚）：冷专家卸载到硬盘（可选，`XTIER_COLD=1`），四个模型都能用；DeepSeek 卸载 30% 时 decode 每步 +44%。
  详见[文档](reports/2026-10-26-cold-experts.md)。
- 2026-10-26：弹性热专家槽（DeepSeek 每步 −4.3%、GLM −1.5%）；GLM 常备槽 60 → 58；启动台报在线前先预热；MiniMax-M3.1-Flash
  暂停开发。详见[文档](reports/2026-10-26-elastic-cache.md)。
- 2026-10-25：测速表改为官方 main 对加速包；Qwen 启动台默认用自动规划布局；热专家池可选用提示的路由预热（`XTIER_CACHE_PROMPT=1`，
  默认关）；修了 MiniMax 包钩子和 GC 冻结两个启动报错。详见[文档](reports/2026-10-25-prompt-warmup.md)。
- 2026-10-24（晚）：自动规划器第一版：本机测出各层 GPU / CPU 耗时，在显存预算内逐层定热专家槽数（`XTIER_CACHE_PLAN=auto`）；
  Qwen 启动时自动布局冷 prefill +2.4–2.9%，其余模型保留手工布局。详见[文档](reports/2026-10-24-planner.md)。
- 2026-10-24（下午）：MiniMax 固定 512K、KV 存 NVFP4（42.5 → 24 GiB），DSpark 按置信度截断验证（无损，agent decode +11%）。详见[文档](reports/2026-10-24-minimax-512k.md)。
- 2026-10-24（上午，二）：SM120 上 prefill 大小的 MXFP8 GEMM 改用 128×128 tile（输出逐位相同），MiniMax 64K 冷 prefill
  1,606 → 1,963 tok/s。详见[文档](reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（上午）：新模型 MiniMax-M3.1-Flash（SGLang M3.1 分支），加速包 agent 每步 70.6 → 59.5 ms；DeepSeek 索引器 decode
  走分页路径，每步 33.0 → 32.5 ms。详见[文档](reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（清晨）：DeepSeek 的 wo_a 保持 FP8（每步 33.9 → 33.0 ms，显存 −1.25 GiB），提示按 deepseek-recipe 分词；修了 TP 1
  下 NCCL 临时申请显存导致的 OOM。详见[文档](reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（夜）：测速统一用 agent 速度；启动参数补上生产的接口参数（缓存命中上报、工具调用解析）。详见[文档](reports/2026-10-24-qwen-kv-ds-agent.md)。
- 2026-10-24：控制台 `xtier/console/`：一个网页选模型、调参数、启动 / 停止，加模型只需加配置目录。详见[文档](console.md)。
- 2026-10-24：Qwen 默认 NVFP4 KV，省下的显存多放一层常驻（冷 prefill +0.5–1.4%）；DeepSeek 按生产 agent_bench 实测冷 prefill 快
  10–25%。详见[文档](reports/2026-10-24-qwen-kv-ds-agent.md)。
- 2026-10-23（夜）：通用投机解码包：GLM 用热词表拒绝采样和置信度截断，decode 93.0 → 95.6 tok/s（生产 84.8）。详见[文档](reports/2026-10-23-speculative-fp4kv.md)。
- 2026-10-23（晚）：512K prefill 追平生产：Qwen 125 → 102 s（生产 106 s），GLM 224 → 192 s（生产 201 s）。详见[文档](reports/2026-10-23-prefill.md)。
- 2026-10-23（晚）：DeepSeek 移植 Engram 提前查表和草稿词表头 FP8，decode 每步 −8%。详见[文档](reports/2026-10-23-ds-engram-prefetch.md)。
- 2026-10-23：README 测速表；加载不再换出，DeepSeek 启动 811 → 409 s。详见[文档](reports/2026-10-23-step3-ds-tables.md)。
- 2026-10-23：第 3 步：DeepSeek 跑通官方 main，通用 NVMe 表，GLM 的 FP8 权重，双卡准备。详见[文档](reports/2026-10-23-step3-ds-tables.md)。
- 2026-10-23：第 2 步：后端、热专家缓存、GPU prefill 在官方 main 上跑通 Qwen 与 GLM。详见[文档](reports/2026-10-23-step2-qwen.md)。
- 2026-10-23：第 1 步：CPU 引擎迁入 `csrc/`，三个模型的真实层与 lkqmoe 0.5.6 逐位一致（`bench/results/engine-ab-20261023.json`）。
