#pragma once
#include "llama-kv-cache.h"
#include "llama-ssd-offload.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

class SsdWriteWorker {
public:
    SsdWriteWorker();
    ~SsdWriteWorker();
    void submit(llama_seq_id seq_id, std::function<void()> fn);
    void wait_seq(llama_seq_id seq_id);
    void drain();
private:
    struct Job { llama_seq_id seq_id; std::function<void()> fn; };
    std::thread              worker_;
    std::mutex               mu_;
    std::condition_variable  cv_;
    std::queue<Job>          queue_;
    bool                     stop_  = false;
    llama_seq_id             cur_seq_ = -1;
    std::condition_variable  done_cv_;
    void run();
};

class llama_kv_evict_mgr {
public:
    llama_kv_evict_mgr(llama_kv_cache & kv, const std::string & ssd_path);
    ~llama_kv_evict_mgr();

    void touch_seq(llama_seq_id seq_id);
    int  try_evict(const std::vector<llama_seq_id> & active_seqs);
    bool ensure_restored(const std::vector<llama_seq_id> & seq_ids);
    bool is_evicted(llama_seq_id seq_id) const;
    void print_stats() const;

private:
    llama_kv_cache &  kv_;
    llama_ssd_stats   ssd_stats_;
    llama_ssd_kv_file ssd_file_;

    struct CellMeta { uint32_t orig_cell; llama_pos pos; };

    struct EvictEntry {
        llama_seq_id seq_id;
        uint32_t     strm;
        off_t        ssd_base_offset;
        size_t       total_bytes;
        std::vector<CellMeta> cells;
    };

    struct PrefetchEntry {
        void *            buf      = nullptr;
        size_t            buf_size = 0;
        std::atomic<bool> ready    {false};
        bool              ok       = false;
        std::function<void(void*)> buf_free = [](void * p){ free(p); };
        ~PrefetchEntry() { if (buf) buf_free(buf); }
    };

    std::unordered_map<llama_seq_id, EvictEntry>                       evicted_map_;
    std::unordered_map<llama_seq_id, uint64_t>                         lru_time_;
    std::unordered_map<llama_seq_id, std::shared_ptr<PrefetchEntry>>   prefetch_map_;
    uint64_t clock_tick_ = 0;

    struct FreeRegion { off_t offset; size_t size; };
    std::vector<FreeRegion> free_list_;
    off_t next_offset_ = 0;

    SsdWriteWorker write_worker_;

    off_t  alloc_space(size_t bytes);
    void   free_space(off_t offset, size_t bytes);

    bool do_evict  (llama_seq_id seq_id);
    bool do_restore(llama_seq_id seq_id);

    llama_seq_id pick_victim(const std::vector<llama_seq_id> & active) const;

    // helper: scatter flat buffer back to KV cache tensors
    void do_scatter(const void * buf,
                    const std::vector<uint32_t> & new_cells,
                    const EvictEntry & entry);

    static bool cells_contiguous(const std::vector<uint32_t> & idxs);
};
