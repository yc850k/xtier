# 版本更新

按日期倒序，README 只列最近 5 条。

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
     BF16——卡上没有 BF16 张量核，这部分是速度的主要代价。
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
