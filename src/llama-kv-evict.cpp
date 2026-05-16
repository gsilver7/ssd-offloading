#include "llama-kv-evict.h"
#include "llama-impl.h"   // for LLAMA_LOG_*
#include "ggml-backend.h" // for ggml_backend_tensor_get/set
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#ifdef GGML_USE_CUDA
#  include <cuda_runtime.h>
#endif

#ifdef LLAMA_SSD_NVTX
#  include <nvToolsExt.h>
#  define NVTX_PUSH(s) nvtxRangePushA(s)
#  define NVTX_POP()   nvtxRangePop()
#else
#  define NVTX_PUSH(s) (void)0
#  define NVTX_POP()   (void)0
#endif

// ─── helpers ────────────────────────────────────────────────────────────────

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double elapsed_ms(double t0) {
    return now_ms() - t0;
}

// ─── SsdWriteWorker ─────────────────────────────────────────────────────────

SsdWriteWorker::SsdWriteWorker() {
    worker_ = std::thread([this]{ run(); });
}

SsdWriteWorker::~SsdWriteWorker() {
    {
        std::unique_lock<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void SsdWriteWorker::submit(llama_seq_id seq_id, std::function<void()> fn) {
    {
        std::unique_lock<std::mutex> lk(mu_);
        queue_.push({seq_id, std::move(fn)});
    }
    cv_.notify_one();
}

void SsdWriteWorker::wait_seq(llama_seq_id seq_id) {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&]{
        // done when: the queue has no job for seq_id AND cur_seq_ != seq_id
        if (cur_seq_ == seq_id) return false;
        for (auto tmp = queue_; !tmp.empty(); tmp.pop()) {
            if (tmp.front().seq_id == seq_id) return false;
        }
        return true;
    });
}

void SsdWriteWorker::drain() {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&]{ return queue_.empty() && cur_seq_ == -1; });
}

void SsdWriteWorker::run() {
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&]{ return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) break;
            job = std::move(queue_.front());
            queue_.pop();
            cur_seq_ = job.seq_id;
        }
        job.fn();
        {
            std::unique_lock<std::mutex> lk(mu_);
            cur_seq_ = -1;
        }
        done_cv_.notify_all();
    }
}

// ─── llama_kv_evict_mgr ─────────────────────────────────────────────────────

llama_kv_evict_mgr::llama_kv_evict_mgr(llama_kv_cache & kv, const std::string & ssd_path)
    : kv_(kv), ssd_file_(ssd_stats_)
{
    if (!ssd_file_.open(ssd_path)) {
        throw std::runtime_error("[llama-ssd] failed to open SSD KV-cache file: " + ssd_path);
    }
    LLAMA_LOG_INFO("[llama-ssd] SSD KV-cache offload enabled: %s\n", ssd_path.c_str());
}

llama_kv_evict_mgr::~llama_kv_evict_mgr() {
    write_worker_.drain();
    ssd_file_.close();
}

// ─── public API ─────────────────────────────────────────────────────────────

void llama_kv_evict_mgr::touch_seq(llama_seq_id seq_id) {
    if (seq_id < 0) return;
    lru_time_[seq_id] = clock_tick_++;
}

bool llama_kv_evict_mgr::is_evicted(llama_seq_id seq_id) const {
    return evicted_map_.count(seq_id) > 0;
}

int llama_kv_evict_mgr::try_evict(const std::vector<llama_seq_id> & active_seqs) {
    llama_seq_id victim = pick_victim(active_seqs);
    if (victim < 0) return 0;
    return do_evict(victim) ? 1 : 0;
}

bool llama_kv_evict_mgr::ensure_restored(const std::vector<llama_seq_id> & seq_ids) {
    bool ok = true;
    for (llama_seq_id sid : seq_ids) {
        if (sid >= 0 && is_evicted(sid)) {
            if (!do_restore(sid)) {
                ok = false;
            }
        }
    }
    return ok;
}

void llama_kv_evict_mgr::print_stats() const {
    ssd_stats_.print_summary();
}

// ─── private helpers ─────────────────────────────────────────────────────────

off_t llama_kv_evict_mgr::alloc_space(size_t bytes) {
    // Best-fit from free_list_
    int best = -1;
    for (int i = 0; i < (int)free_list_.size(); ++i) {
        if (free_list_[i].size >= bytes) {
            if (best < 0 || free_list_[i].size < free_list_[best].size) {
                best = i;
            }
        }
    }
    if (best >= 0) {
        off_t off = free_list_[best].offset;
        free_list_[best].offset += (off_t)bytes;
        free_list_[best].size   -= bytes;
        if (free_list_[best].size == 0) {
            free_list_.erase(free_list_.begin() + best);
        }
        return off;
    }
    off_t off = next_offset_;
    next_offset_ += (off_t)bytes;
    return off;
}

void llama_kv_evict_mgr::free_space(off_t offset, size_t bytes) {
    free_list_.push_back({offset, bytes});
    std::sort(free_list_.begin(), free_list_.end(),
              [](const FreeRegion & a, const FreeRegion & b){ return a.offset < b.offset; });
    // merge adjacent
    for (size_t i = 1; i < free_list_.size(); ) {
        FreeRegion & prev = free_list_[i-1];
        FreeRegion & cur  = free_list_[i];
        if (prev.offset + (off_t)prev.size == cur.offset) {
            prev.size += cur.size;
            free_list_.erase(free_list_.begin() + (int)i);
        } else {
            ++i;
        }
    }
}

bool llama_kv_evict_mgr::cells_contiguous(const std::vector<uint32_t> & idxs) {
    if (idxs.empty()) return true;
    for (size_t j = 1; j < idxs.size(); ++j) {
        if (idxs[j] != idxs[0] + (uint32_t)j) return false;
    }
    return true;
}

llama_seq_id llama_kv_evict_mgr::pick_victim(const std::vector<llama_seq_id> & active) const {
    llama_seq_id best = -1;
    uint64_t     best_time = UINT64_MAX;

    for (const auto & [sid, t] : lru_time_) {
        // skip if active
        if (std::find(active.begin(), active.end(), sid) != active.end()) continue;
        // skip if already evicted
        if (is_evicted(sid)) continue;
        // skip if no cells
        uint32_t strm = kv_.get_seq_stream(sid);
        if (kv_.get_seq_cell_indices(sid, strm).empty()) continue;

        if (t < best_time) {
            best_time = t;
            best = sid;
        }
    }
    return best;
}

// ─── do_scatter ──────────────────────────────────────────────────────────────

void llama_kv_evict_mgr::do_scatter(const void * buf,
                                     const std::vector<uint32_t> & new_cells,
                                     const EvictEntry & entry)
{
    const size_t  n_cells  = new_cells.size();
    const char *  p        = static_cast<const char *>(buf);
    const bool    opt_slab = (std::getenv("LLAMA_SSD_OPT_SLAB") != nullptr);
    const bool    contig   = opt_slab && cells_contiguous(new_cells);

    for (int32_t il : kv_.get_kv_model_layer_ids()) {
        // K
        auto * kt    = kv_.get_k_tensor(il, entry.strm);
        size_t k_cell = kv_.get_k_cell_bytes(il);
        size_t k_slab = n_cells * k_cell;
        if (kt) {
            if (contig) {
                ggml_backend_tensor_set(kt, p, new_cells[0]*k_cell, k_slab);
            } else {
                for (size_t j = 0; j < n_cells; ++j) {
                    ggml_backend_tensor_set(kt, p + j*k_cell, new_cells[j]*k_cell, k_cell);
                }
            }
        }
        p += k_slab;

        // V
        auto * vt    = kv_.get_v_tensor(il, entry.strm);
        size_t v_cell = kv_.get_v_cell_bytes(il);
        size_t v_slab = n_cells * v_cell;
        if (vt) {
            if (contig) {
                ggml_backend_tensor_set(vt, p, new_cells[0]*v_cell, v_slab);
            } else {
                for (size_t j = 0; j < n_cells; ++j) {
                    ggml_backend_tensor_set(vt, p + j*v_cell, new_cells[j]*v_cell, v_cell);
                }
            }
        }
        p += v_slab;
    }
}

// ─── do_evict ────────────────────────────────────────────────────────────────

bool llama_kv_evict_mgr::do_evict(llama_seq_id seq_id) {
    // read env opts
    const bool opt_slab        = (std::getenv("LLAMA_SSD_OPT_SLAB")    != nullptr);
    const bool opt_pinned      = (std::getenv("LLAMA_SSD_OPT_PINNED")  != nullptr);
#ifdef GGML_USE_CUDA
    const bool opt_batch_async = (std::getenv("LLAMA_SSD_OPT_BATCH")   != nullptr);
#endif
    const bool opt_prefetch    = (std::getenv("LLAMA_SSD_OPT_PREFETCH")!= nullptr);

    const double t_total_start = now_ms();

    uint32_t strm = kv_.get_seq_stream(seq_id);
    std::vector<uint32_t> cell_idxs = kv_.get_seq_cell_indices(seq_id, strm);
    const size_t n_cells = cell_idxs.size();
    if (n_cells == 0) return false;

    // compute total bytes
    size_t total_bytes = 0;
    for (int32_t il : kv_.get_kv_model_layer_ids()) {
        total_bytes += n_cells * kv_.get_k_cell_bytes(il);
        total_bytes += n_cells * kv_.get_v_cell_bytes(il);
    }

    off_t base = alloc_space(total_bytes);

    // build entry metadata
    EvictEntry entry;
    entry.seq_id          = seq_id;
    entry.strm            = strm;
    entry.ssd_base_offset = base;
    entry.total_bytes     = total_bytes;
    for (uint32_t idx : cell_idxs) {
        entry.cells.push_back({idx, kv_.get_cell_pos(idx, strm)});
    }

    // ─── OPT2: BATCH_ASYNC path (CUDA only) ──────────────────────────────────
#ifdef GGML_USE_CUDA
    if (opt_batch_async) {
        void * dev_stage  = nullptr;
        void * batch_host = nullptr;
        bool   batch_ok   = false;

        if (cudaMalloc(&dev_stage, total_bytes) == cudaSuccess &&
            cudaMallocHost(&batch_host, total_bytes) == cudaSuccess)
        {
            cudaStream_t cs;
            cudaStreamCreate(&cs);
            NVTX_PUSH("EVICT[BATCH] D2D");
            size_t blob_off = 0;
            for (int32_t il : kv_.get_kv_model_layer_ids()) {
                auto * kt   = kv_.get_k_tensor(il, strm);
                size_t k_cell = kv_.get_k_cell_bytes(il);
                size_t k_slab = n_cells * k_cell;
                if (kt && kt->data) {
                    if (opt_slab && cells_contiguous(cell_idxs)) {
                        cudaMemcpyAsync((char*)dev_stage + blob_off,
                                        (char*)kt->data + cell_idxs[0]*k_cell,
                                        k_slab, cudaMemcpyDeviceToDevice, cs);
                    } else {
                        for (size_t j = 0; j < n_cells; ++j) {
                            cudaMemcpyAsync((char*)dev_stage + blob_off + j*k_cell,
                                            (char*)kt->data + cell_idxs[j]*k_cell,
                                            k_cell, cudaMemcpyDeviceToDevice, cs);
                        }
                    }
                }
                blob_off += k_slab;

                auto * vt   = kv_.get_v_tensor(il, strm);
                size_t v_cell = kv_.get_v_cell_bytes(il);
                size_t v_slab = n_cells * v_cell;
                if (vt && vt->data) {
                    if (opt_slab && cells_contiguous(cell_idxs)) {
                        cudaMemcpyAsync((char*)dev_stage + blob_off,
                                        (char*)vt->data + cell_idxs[0]*v_cell,
                                        v_slab, cudaMemcpyDeviceToDevice, cs);
                    } else {
                        for (size_t j = 0; j < n_cells; ++j) {
                            cudaMemcpyAsync((char*)dev_stage + blob_off + j*v_cell,
                                            (char*)vt->data + cell_idxs[j]*v_cell,
                                            v_cell, cudaMemcpyDeviceToDevice, cs);
                        }
                    }
                }
                blob_off += v_slab;
            }
            NVTX_POP();
            NVTX_PUSH("EVICT[BATCH] D2H");
            cudaMemcpyAsync(batch_host, dev_stage, total_bytes, cudaMemcpyDeviceToHost, cs);
            cudaStreamSynchronize(cs);
            cudaStreamDestroy(cs);
            cudaFree(dev_stage);
            NVTX_POP();

            // write slice-by-slice
            const double t_ssd_start = now_ms();
            NVTX_PUSH("EVICT[BATCH] ssd_write");
            {
                size_t off2 = 0;
                for (int32_t il : kv_.get_kv_model_layer_ids()) {
                    size_t k_slab = n_cells * kv_.get_k_cell_bytes(il);
                    size_t v_slab = n_cells * kv_.get_v_cell_bytes(il);
                    ssd_file_.write_kv((const char*)batch_host + off2, k_slab, base + (off_t)off2);
                    off2 += k_slab;
                    ssd_file_.write_kv((const char*)batch_host + off2, v_slab, base + (off_t)off2);
                    off2 += v_slab;
                }
            }
            NVTX_POP();
            const double t_ssd_ms = elapsed_ms(t_ssd_start);

            cudaFreeHost(batch_host);
            batch_ok = true;

            kv_.seq_rm(seq_id, -1, -1);
            evicted_map_[seq_id] = entry;

            const double t_total_ms = elapsed_ms(t_total_start);
            LLAMA_LOG_INFO("[BENCH] op=EVICT seq=%d bytes=%zu ssd=%.1fms total=%.1fms opts=OPT2-BATCH\n",
                           (int)seq_id, total_bytes, t_ssd_ms, t_total_ms);

            if (opt_prefetch) {
                auto pf = std::make_shared<PrefetchEntry>();
                pf->buf_size = total_bytes;
                void * pf_buf = nullptr;
                bool   pf_pinned = false;
#ifdef GGML_USE_CUDA
                if (cudaMallocHost(&pf_buf, pf->buf_size) == cudaSuccess) {
                    pf_pinned = true;
                    pf->buf_free = [](void*p){ cudaFreeHost(p); };
                }
#endif
                if (!pf_buf) {
                    const size_t pf_aligned = (pf->buf_size + 4095) & ~(size_t)4095;
                    if (posix_memalign(&pf_buf, 4096, pf_aligned) != 0) { pf_buf = nullptr; }
                }
                pf->buf = pf_buf;
                (void)pf_pinned;
                prefetch_map_[seq_id] = pf;

                const off_t  read_offset = base;
                const size_t read_size   = total_bytes;
                write_worker_.submit(seq_id, [this, pf, read_offset, read_size]() {
                    pf->ok = ssd_file_.read_kv(pf->buf, read_size, read_offset);
                    pf->ready.store(true, std::memory_order_release);
                });
            }

            return true;
        }
        // fallthrough to baseline if alloc failed
        if (dev_stage)  cudaFree(dev_stage);
        if (batch_host) cudaFreeHost(batch_host);
        (void)batch_ok;
    }
#endif // GGML_USE_CUDA

    // ─── BASELINE path ────────────────────────────────────────────────────────
    // Phase 1: GPU → CPU
    const double t_gpu2cpu_start = now_ms();
    NVTX_PUSH("EVICT gpu2cpu");

    // Allocate a flat host buffer for all K+V data
    void * flat_buf = nullptr;
    size_t aligned_total = (total_bytes + 4095) & ~(size_t)4095;
    bool   flat_pinned   = false;
#ifdef GGML_USE_CUDA
    if (opt_pinned) {
        if (cudaMallocHost(&flat_buf, aligned_total) == cudaSuccess) {
            flat_pinned = true;
        }
    }
#endif
    if (!flat_buf) {
        if (posix_memalign(&flat_buf, 4096, aligned_total) != 0) { flat_buf = nullptr; }
    }

    char * wp = static_cast<char *>(flat_buf);
    for (int32_t il : kv_.get_kv_model_layer_ids()) {
        // K
        auto * kt    = kv_.get_k_tensor(il, strm);
        size_t k_cell = kv_.get_k_cell_bytes(il);
        size_t k_slab = n_cells * k_cell;
        if (kt) {
            if (opt_slab && cells_contiguous(cell_idxs)) {
                ggml_backend_tensor_get(kt, wp, cell_idxs[0]*k_cell, k_slab);
            } else {
                for (size_t j = 0; j < n_cells; ++j) {
                    ggml_backend_tensor_get(kt, wp + j*k_cell, cell_idxs[j]*k_cell, k_cell);
                }
            }
        }
        wp += k_slab;

        // V
        auto * vt    = kv_.get_v_tensor(il, strm);
        size_t v_cell = kv_.get_v_cell_bytes(il);
        size_t v_slab = n_cells * v_cell;
        if (vt) {
            if (opt_slab && cells_contiguous(cell_idxs)) {
                ggml_backend_tensor_get(vt, wp, cell_idxs[0]*v_cell, v_slab);
            } else {
                for (size_t j = 0; j < n_cells; ++j) {
                    ggml_backend_tensor_get(vt, wp + j*v_cell, cell_idxs[j]*v_cell, v_cell);
                }
            }
        }
        wp += v_slab;
    }

    NVTX_POP();
    const double t_gpu2cpu_ms = elapsed_ms(t_gpu2cpu_start);

    // Phase 2: write to SSD
    const double t_ssd_start = now_ms();
    NVTX_PUSH("EVICT ssd_write");
    bool write_ok = ssd_file_.write_kv(flat_buf, total_bytes, base);
    NVTX_POP();
    const double t_ssd_ms = elapsed_ms(t_ssd_start);

    if (flat_pinned) {
#ifdef GGML_USE_CUDA
        cudaFreeHost(flat_buf);
#endif
    } else {
        free(flat_buf);
        flat_buf = nullptr;
    }

    if (!write_ok) {
        free_space(base, total_bytes);
        return false;
    }

    kv_.seq_rm(seq_id, -1, -1);
    evicted_map_[seq_id] = entry;

    // Build opts string
    std::string opts_str;
    if (opt_slab)    { if (!opts_str.empty()) opts_str+=","; opts_str+="SLAB"; }
    if (opt_pinned)  { if (!opts_str.empty()) opts_str+=","; opts_str+="PINNED"; }
    if (opt_prefetch){ if (!opts_str.empty()) opts_str+=","; opts_str+="PREFETCH"; }
    if (opts_str.empty()) opts_str = "BASELINE";

    const double t_total_ms = elapsed_ms(t_total_start);
    LLAMA_LOG_INFO("[BENCH] op=EVICT seq=%d bytes=%zu gpu2cpu=%.1fms ssd=%.1fms total=%.1fms opts=%s\n",
                   (int)seq_id, total_bytes, t_gpu2cpu_ms, t_ssd_ms, t_total_ms, opts_str.c_str());

    // OPT4: prefetch-read back into pinned buffer for the next restore
    if (opt_prefetch) {
        auto pf = std::make_shared<PrefetchEntry>();
        pf->buf_size = total_bytes;
        void * pf_buf = nullptr;
        bool   pf_pinned = false;
#ifdef GGML_USE_CUDA
        if (cudaMallocHost(&pf_buf, aligned_total) == cudaSuccess) {
            pf_pinned = true;
            pf->buf_free = [](void*p){ cudaFreeHost(p); };
        }
#endif
        if (!pf_buf) {
            if (posix_memalign(&pf_buf, 4096, aligned_total) != 0) { pf_buf = nullptr; }
        }
        pf->buf = pf_buf;
        (void)pf_pinned;
        prefetch_map_[seq_id] = pf;

        const off_t  read_offset = base;
        const size_t read_size   = total_bytes;
        write_worker_.submit(seq_id, [this, pf, read_offset, read_size]() {
            pf->ok = ssd_file_.read_kv(pf->buf, read_size, read_offset);
            pf->ready.store(true, std::memory_order_release);
        });
    }

    return true;
}

// ─── do_restore ──────────────────────────────────────────────────────────────

bool llama_kv_evict_mgr::do_restore(llama_seq_id seq_id) {
    // read env opts
    const bool opt_blob_read  = (std::getenv("LLAMA_SSD_OPT_BLOB")    != nullptr);
    const bool opt_prefetch   = (std::getenv("LLAMA_SSD_OPT_PREFETCH")!= nullptr);
    const bool opt_slab       = (std::getenv("LLAMA_SSD_OPT_SLAB")    != nullptr);

    const double t_total_start = now_ms();

    auto it_e = evicted_map_.find(seq_id);
    if (it_e == evicted_map_.end()) return false;
    EvictEntry & entry = it_e->second;

    // Collect original positions
    std::vector<llama_pos> positions;
    positions.reserve(entry.cells.size());
    for (const auto & cm : entry.cells) {
        positions.push_back(cm.pos);
    }

    // Claim cells in the KV cache
    std::vector<uint32_t> new_cells = kv_.claim_cells(seq_id, positions, entry.strm);
    if (new_cells.empty()) {
        return false;
    }

    (void)new_cells.size(); // n_cells used inside OPT4/OPT5/BASELINE blocks via do_scatter

    // ─── OPT4: prefetch path ─────────────────────────────────────────────────
    if (opt_prefetch) {
        write_worker_.wait_seq(seq_id);
        auto it_pf = prefetch_map_.find(seq_id);
        if (it_pf != prefetch_map_.end()) {
            auto pf = it_pf->second;
            prefetch_map_.erase(it_pf);

            NVTX_PUSH("RESTORE[PREFETCH] wait");
            while (!pf->ready.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            NVTX_POP();

            if (pf->ok) {
                const double t_scatter_start = now_ms();
                NVTX_PUSH("RESTORE[PREFETCH] scatter");
                do_scatter(pf->buf, new_cells, entry);
                NVTX_POP();
                const double t_scatter_ms = elapsed_ms(t_scatter_start);

                free_space(entry.ssd_base_offset, entry.total_bytes);
                evicted_map_.erase(seq_id);
                touch_seq(seq_id);

                const double t_total_ms = elapsed_ms(t_total_start);
                LLAMA_LOG_INFO("[BENCH] op=RESTORE seq=%d bytes=%zu scatter=%.1fms total=%.1fms opts=OPT4-PREFETCH\n",
                               (int)seq_id, entry.total_bytes, t_scatter_ms, t_total_ms);
                return true;
            }
            // prefetch read failed — fall through to baseline
        }
    }

    // ─── OPT5: blob read path ────────────────────────────────────────────────
    if (opt_blob_read) {
        const size_t total    = entry.total_bytes;
        const size_t aligned  = (total + 4095) & ~(size_t)4095;
        void *       blob     = nullptr;
        if (posix_memalign(&blob, 4096, aligned) != 0 || !blob) return false;

        const double t_ssd_start = now_ms();
        NVTX_PUSH("RESTORE[BLOB] ssd_read");
        bool ok = ssd_file_.read_blob(blob, total, entry.ssd_base_offset);
        NVTX_POP();
        const double t_ssd_ms = elapsed_ms(t_ssd_start);

        if (!ok) { free(blob); return false; }

        NVTX_PUSH("RESTORE[BLOB] scatter");
        do_scatter(blob, new_cells, entry);
        NVTX_POP();
        free(blob);

        free_space(entry.ssd_base_offset, entry.total_bytes);
        const size_t logged_bytes = entry.total_bytes;
        evicted_map_.erase(seq_id);
        touch_seq(seq_id);

        const double t_total_ms = elapsed_ms(t_total_start);
        std::string opts_str = "OPT5-BLOB";
        if (opt_slab) opts_str += ",SLAB";
        LLAMA_LOG_INFO("[BENCH] op=RESTORE seq=%d bytes=%zu ssd=%.1fms total=%.1fms opts=%s\n",
                       (int)seq_id, logged_bytes, t_ssd_ms, t_total_ms, opts_str.c_str());
        return true;
    }

    // ─── BASELINE: per-layer read ─────────────────────────────────────────────
    {
        const double t_ssd_start = now_ms();
        NVTX_PUSH("RESTORE ssd_read");

        // Read all data into a flat buffer, then scatter
        const size_t total   = entry.total_bytes;
        const size_t aligned = (total + 4095) & ~(size_t)4095;
        void * flat_buf = nullptr;
        if (posix_memalign(&flat_buf, 4096, aligned) != 0 || !flat_buf) { return false; }

        bool ok = ssd_file_.read_kv(flat_buf, total, entry.ssd_base_offset);
        NVTX_POP();
        const double t_ssd_ms = elapsed_ms(t_ssd_start);

        if (!ok) { free(flat_buf); return false; }

        NVTX_PUSH("RESTORE scatter");
        do_scatter(flat_buf, new_cells, entry);
        NVTX_POP();
        free(flat_buf);

        free_space(entry.ssd_base_offset, entry.total_bytes);
        const size_t logged_bytes = entry.total_bytes;
        evicted_map_.erase(seq_id);
        touch_seq(seq_id);

        const double t_total_ms = elapsed_ms(t_total_start);
        std::string opts_str;
        if (opt_slab) opts_str = "SLAB";
        if (opts_str.empty()) opts_str = "BASELINE";
        LLAMA_LOG_INFO("[BENCH] op=RESTORE seq=%d bytes=%zu ssd=%.1fms total=%.1fms opts=%s\n",
                       (int)seq_id, logged_bytes, t_ssd_ms, t_total_ms, opts_str.c_str());
        return true;
    }
}
