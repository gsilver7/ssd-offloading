#pragma once
#include <string>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <sys/types.h>

struct llama_ssd_stats {
    std::atomic<uint64_t> read_bytes {0};
    std::atomic<uint64_t> write_bytes{0};
    std::atomic<uint64_t> read_ops   {0};
    std::atomic<uint64_t> write_ops  {0};

    void print_summary() const;
};

struct llama_ssd_kv_file {
    bool open(const std::string & path);
    void close();
    bool is_open() const { return fd >= 0; }
    bool write_kv(const void * src, size_t nbytes, off_t offset);
    bool read_kv (void * dst,       size_t nbytes, off_t offset);
    bool read_blob(void * dst,      size_t nbytes, off_t offset); // OPT5: single blob read

    llama_ssd_stats & stats;  // reference to shared stats
    explicit llama_ssd_kv_file(llama_ssd_stats & s) : stats(s) {}

private:
    int fd = -1;
};
