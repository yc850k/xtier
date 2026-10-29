"""The cold store's reads of the GSQ experts (the checkpoint's own GGUF shards): the O_DIRECT path against the
buffered one, the drive's bandwidth and the bytes it moved.

  python tests/gsq_cold_check.py

The GSQ path does not copy the experts into the store: it reads the checkpoint's own tensors, whose offsets are 64
bytes modulo 512 (the drives' logical block), so the store cannot read them straight into the destination (O_DIRECT)
and reads the aligned window around the range instead. The store's files are the checkpoint's shards on /mnt/optane (a
Dell Ent NVMe P5800x on ext4, which supports O_DIRECT): the measurement's drive.

Measured: the bytes of one expert against the same range read with the buffered path, the bandwidth of one expert and
of a burst of experts (the drive's parallelism), the latency of one expert, the bytes the drive moved (/proc/diskstats)
against the bytes asked for, and the page cache's growth (with O_DIRECT: none).
"""
import argparse
import ctypes
import mmap
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xtier.backend import gsq          # noqa: E402
from xtier.engine import ColdStore     # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--pack', default='/mnt/optane/DeepSeek-V4.1-Flash/maya-ds4-gsq',
               help="the GSQ pack (the checkpoint's shards are beside it)")
P.add_argument('--layer', type=int, default=0)
P.add_argument('--expert', type=int, default=0)
P.add_argument('--burst', type=int, default=32, help="experts read at once (the drive's parallelism)")
P.add_argument('--threads', type=int, default=8, help="the store's readers per file")
P.add_argument('--repeats', type=int, default=5, help='the reads of one expert (the latency)')
A = P.parse_args()

ROLES = ('w1', 'w3', 'w2')      # the gate, the up, the down

bad = []


def check(ok, what):
    print(('ok   ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        bad.append(what)


def destination(size):
    """A page-aligned destination buffer (the store reads into it). Its pages are touched, so what the page cache
    grows by is the store's file data and not the buffers."""
    mapped = mmap.mmap(-1, size)
    mapped[::4096] = b'\0' * ((size + 4095) // 4096)
    return mapped, ctypes.addressof(ctypes.c_char.from_buffer(mapped))


def diskstats(names):
    """The bytes the drives read (the sectors read of the devices)."""
    sectors = {}
    for line in open('/proc/diskstats'):
        f = line.split()
        if f[2] in names:
            sectors[f[2]] = int(f[5])
    return sum(sectors.values()) * 512


def cached_bytes():
    """The page cache's size."""
    for line in open('/proc/meminfo'):
        if line.startswith('Cached:'):
            return int(line.split()[1]) * 1024


def device(path):
    """The diskstats name of the partition holding the file (the disk's line holds every partition)."""
    st = os.stat(path)
    return os.path.basename(os.path.realpath(f'/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}'))


def evict(path, ranges):
    """The ranges out of the page cache (the buffered read below is a cold one)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        for off, size in ranges:
            os.posix_fadvise(fd, off, size, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def mount_of(path):
    """The mount point of a path (the store's files must be on a filesystem that supports O_DIRECT)."""
    best = ''
    for line in open('/proc/mounts'):
        parts = line.split()
        if len(parts) > 2 and parts[1].startswith('/') and str(path).startswith(parts[1]):
            if len(parts[1]) > len(best):
                best = parts[1]
    return best


def gbs(count, seconds):
    return count / 1e9 / seconds if seconds > 0 else 0.0


pack = gsq.GsqPack(A.pack)
# the shards holding the experts, and the layer of every shard (its experts): the burst spreads over the shards
shards = sorted({pack.tensors[f'layers.{l}.ffn.experts.0.w1.weight'][1] for l in range(pack.layers)})
paths = [pack.file_path(f) for f in shards]
index = {f: i for i, f in enumerate(shards)}
devices = sorted({device(p) for p in paths})
layer_of = {}
for l in range(pack.layers):
    layer_of.setdefault(pack.tensors[f'layers.{l}.ffn.experts.0.w1.weight'][1], l)
burst_layers = [layer_of[f] for f in shards]
locations = {l: pack.expert_locations(l) for l in set(burst_layers) | {A.layer}}

store = ColdStore(paths, threads=A.threads, writable=False)


def tensors(layer, expert):
    """One expert's tensors: (store file, offset, bytes, role)."""
    return [(index[f], off, size, role) for role in ROLES
            for f, off, size, _ in (locations[layer][expert][role],)]


def stored(f, off, size):
    """The store's read of a range: the bytes."""
    buf, addr = destination(size)
    store.read([(f, off, [(addr, size)])])
    return bytes(buf)


def buffered(f, off, size):
    """The same range read with the page cache (the ordinary path)."""
    with open(paths[f], 'rb') as fh:
        fh.seek(off)
        return fh.read(size)


# the read path: a range that is not whole blocks (the GSQ tensors), one that is (the FP4 path's records, read
# straight into the destination) and one at the end of the file (the window reaches past it: the read is short)
shard_size = os.stat(paths[0]).st_size
for off, size, what in (((shard_size // 2) & ~511, 1 << 20, 'an aligned range'),
                         (shard_size - (1 << 20), 1 << 20, 'a range at the end of the file'),
                         ((shard_size // 2) + 64, 1 << 20, 'a range at 64 bytes modulo 512')):
    check(stored(0, off, size) == buffered(0, off, size),
          f'{what}: {size} bytes at {off} (offset % 512 = {off % 512}) match the buffered read')

# the FP4 path: a record written and read back (whole blocks: straight into the destination, no copy)
with tempfile.TemporaryDirectory(dir=mount_of(paths[0])) as tmp:
    fp4 = ColdStore([str(Path(tmp) / 'fp4.cold')], threads=2, writable=True)
    record = bytes(range(256)) * 4096                       # 1 MiB
    src, src_addr = destination(len(record))
    src.write(record)
    fp4.write(0, 0, [(src_addr, len(record))])
    back, back_addr = destination(len(record))
    fp4.read([(0, 0, [(back_addr, len(record))])])
    check(bytes(back) == record, 'the FP4 path: a record written and read back (whole blocks)')
    fp4.close()

# one expert: the store's O_DIRECT read against the buffered read of the same ranges
expert = A.expert
one = tensors(A.layer, expert)
bufs = {role: destination(size) for _, _, size, role in one}
store.read([(f, off, [(bufs[role][1], size)]) for f, off, size, role in one])
one_bytes = sum(size for _, _, size, _ in one)
for f, off, size, role in one:
    with open(paths[f], 'rb') as fh:
        fh.seek(off)
        want = fh.read(size)
    check(bytes(bufs[role][0]) == want,
          f'layer {A.layer} expert {expert} {role}: {size} bytes at {off} (offset % 512 = {off % 512}) '
          f'match the buffered read')
print(f'expert {expert} of layer {A.layer}: {one_bytes / 1e6:.1f} MB in {len(one)} tensors '
      f'(the offsets % 512: {[off % 512 for _, off, _, _ in one]})', flush=True)

# the same expert again and again: the latency of one expert (O_DIRECT: the drive every time)
times = []
for _ in range(A.repeats):
    t = time.perf_counter()
    store.read([(f, off, [(bufs[role][1], size)]) for f, off, size, role in one])
    times.append(time.perf_counter() - t)
latency = sum(times) / len(times)

# a burst of experts, spread over the shards: the drive's parallelism
burst = [(burst_layers[i % len(burst_layers)], i // len(burst_layers)) for i in range(A.burst)]
burst_bufs, burst_jobs, burst_bytes = {}, [], 0
burst_ranges = {}
for layer, expert_i in burst:
    for f, off, size, role in tensors(layer, expert_i):
        key = (layer, expert_i, role)
        if key not in burst_bufs:
            burst_bufs[key] = destination(size)
        burst_jobs.append((f, off, [(burst_bufs[key][1], size)]))
        burst_ranges.setdefault(f, []).append((off, size))
        burst_bytes += size
for f, ranges in burst_ranges.items():   # the burst's ranges are not in the page cache
    evict(paths[f], ranges)

disk0, cache0 = diskstats(devices), cached_bytes()
t = time.perf_counter()
store.read(burst_jobs)
burst_s = time.perf_counter() - t
disk = diskstats(devices) - disk0
grew = cached_bytes() - cache0
stats = store.stats()

# the same ranges with the buffered path (the page cache: the drive's bytes are the same, the latency is not)
disk0, cache0 = diskstats(devices), cached_bytes()
t = time.perf_counter()
for f, off, size, role in [t for layer, expert_i in burst for t in tensors(layer, expert_i)]:
    with open(paths[f], 'rb') as fh:
        fh.seek(off)
        fh.read(size)
buffered_s = time.perf_counter() - t
buffered_disk = diskstats(devices) - disk0
buffered_grew = cached_bytes() - cache0

print(f'\none expert ({one_bytes / 1e6:.1f} MB): {gbs(one_bytes, latency):.2f} GB/s, {latency * 1e3:.2f} ms', flush=True)
print(f'burst of {len(burst)} experts ({burst_bytes / 1e6:.1f} MB, {len(burst_jobs)} reads over '
      f'{len(burst_layers)} shards): {gbs(burst_bytes, burst_s):.2f} GB/s, {burst_s * 1e3:.1f} ms', flush=True)
print(f'the drive moved {disk / 1e6:.1f} MB for {burst_bytes / 1e6:.1f} MB asked for '
      f'({disk / burst_bytes:.3f}x; the windows add the blocks around the ranges)', flush=True)
print(f'the page cache grew {grew / 1e6:.2f} MB (O_DIRECT: nothing is cached)', flush=True)
print(f'the buffered path over the same ranges: {gbs(burst_bytes, buffered_s):.2f} GB/s, '
      f'the drive moved {buffered_disk / 1e6:.1f} MB, the page cache grew {buffered_grew / 1e6:.1f} MB', flush=True)
print(f'the store: {stats[0] / 1e6:.1f} MB read in {stats[2]:.0f} reads, {stats[3]:.2f} s in reads', flush=True)
print(f'the store\'s files: {len(paths)} shards on {mount_of(paths[0])} (the drive {devices[0]})', flush=True)

check(abs(grew) < 4e6, f'the page cache did not grow with O_DIRECT ({grew / 1e6:.2f} MB)')
check(disk >= burst_bytes, f'the drive moved at least the bytes asked for ({disk / 1e6:.1f} MB)')
check(disk < burst_bytes * 1.25, 'the drive moved no more than the windows around the ranges '
                                 f'({disk / burst_bytes:.3f}x; the drive\'s other users are in this number too)')

sys.exit(1 if bad else 0)
