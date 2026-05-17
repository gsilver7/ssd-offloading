# SSD KV-cache Offload Benchmark Results

**Platform**: Jetson Orin (aarch64), CUDA compute 8.7, 7607 MiB unified VRAM  
**Model**: gemma-3-4b-it-Q4_K_M (2.31 GiB, Q4_K_M)  
**Context**: n_ctx=512, n_seq_max=8, offload_kqv=true, n_gpu_layers=99  
**KV data per seq**: 61440 bytes (base cache, 5 non-SWA layers)  
**Profiling**: nsys 2024.5.4, trace=cuda,nvtx,osrt → `ssd-bench.nsys-rep`

## Timing Results

| Config | EVICT gpu2cpu | EVICT ssd | EVICT total | RESTORE ssd | RESTORE total |
|---|---|---|---|---|---|
| BASELINE      | 2.6 ms | 5.9 ms |  **8.6 ms** | 1.2 ms |  **2.2 ms** |
| OPT1_SLAB     | 0.8 ms | 4.3 ms |  **5.1 ms** | 1.5 ms |  **2.1 ms** |
| OPT2_BATCH    |   —    |39.4 ms |**108.9 ms** | 1.3 ms |  **2.6 ms** |
| OPT3_PINNED   | 2.2 ms | 4.9 ms | **69.6 ms** | 1.5 ms |  **2.8 ms** |
| OPT4_PREFETCH | 2.3 ms | 4.8 ms |  **7.2 ms** | scatter 1.3 ms | **3.2 ms** |
| OPT5_BLOB     | 2.1 ms | 4.9 ms |  **7.0 ms** | 1.1 ms |  **2.5 ms** |

## Combined (EVICT + RESTORE)

| Config | Total | vs BASELINE |
|---|---|---|
| OPT1_SLAB     |  7.2 ms | -33% ✅ |
| OPT5_BLOB     |  9.5 ms | -12% |
| OPT4_PREFETCH | 10.4 ms |  -4% |
| BASELINE      | 10.8 ms | — |
| OPT3_PINNED   | 72.4 ms | +570% ❌ |
| OPT2_BATCH    |111.5 ms |+932% ❌ |

## Analysis

- **OPT1_SLAB**: Best overall. Slab allocator eliminates malloc/free fragmentation,
  reducing both write and read overhead. EVICT 40% faster than BASELINE.

- **OPT5_BLOB**: Contiguous single-pwrite reduces syscall count; second best for EVICT.
  RESTORE uses the standard path so no advantage there.

- **OPT4_PREFETCH**: During EVICT, asynchronously pre-reads data from SSD so RESTORE
  just scatters from the pre-loaded buffer (no SSD read latency). Beneficial when
  evict→restore interval is long; in this back-to-back test the prefetch overlap is limited.

- **OPT3_PINNED**: SSD write itself is fast (4.9 ms) but page-pinning cost dominates
  (~65 ms overhead). On Jetson unified memory, pinning pages is expensive — avoid.

- **OPT2_BATCH**: Async batch write incurs ~70 ms synchronization wait on top of the
  39 ms SSD time — worst total. Not suitable for this workload size.

## Recommendation

| Use case | Recommended |
|---|---|
| General / default | **OPT1_SLAB** |
| Large contiguous payloads | OPT5_BLOB |
| Long evict→restore gap | OPT4_PREFETCH |
| Jetson / unified memory | Avoid OPT3_PINNED, OPT2_BATCH |
