#include "llama-ssd-offload.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cerrno>

// ─── llama_ssd_stats ────────────────────────────────────────────────────────

void llama_ssd_stats::print_summary() const {
    const uint64_t rb = read_bytes .load(std::memory_order_relaxed);
    const uint64_t wb = write_bytes.load(std::memory_order_relaxed);
    const uint64_t ro = read_ops   .load(std::memory_order_relaxed);
    const uint64_t wo = write_ops  .load(std::memory_order_relaxed);

    fprintf(stderr,
        "[SSD-offload] stats: write_ops=%llu write_bytes=%.2f MiB"
        " | read_ops=%llu read_bytes=%.2f MiB\n",
        (unsigned long long)wo, (double)wb / (1024.0*1024.0),
        (unsigned long long)ro, (double)rb / (1024.0*1024.0));
}

// ─── llama_ssd_kv_file ──────────────────────────────────────────────────────

bool llama_ssd_kv_file::open(const std::string & path) {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }

    int flags = O_RDWR | O_CREAT;
#ifdef O_DIRECT
    flags |= O_DIRECT;
#endif

    fd = ::open(path.c_str(), flags, 0644);

    if (fd < 0) {
#ifdef O_DIRECT
        if (errno == EINVAL) {
            // O_DIRECT not supported on this filesystem — fall back to buffered I/O
            flags &= ~O_DIRECT;
            fd = ::open(path.c_str(), flags, 0644);
        }
#endif
        if (fd < 0) {
            perror("[llama-ssd] open");
            return false;
        }
    }

    return true;
}

void llama_ssd_kv_file::close() {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

// Helper: check whether ptr and size are 4096-byte aligned (required for O_DIRECT)
static bool is_aligned(const void * ptr, size_t nbytes) {
    return ((uintptr_t)ptr % 4096 == 0) && (nbytes % 4096 == 0);
}

bool llama_ssd_kv_file::write_kv(const void * src, size_t nbytes, off_t offset) {
    if (fd < 0) return false;
    if (nbytes == 0) return true;

    const void * buf = src;
    void *       staging = nullptr;
    size_t       aligned_size = (nbytes + 4095) & ~(size_t)4095;

    if (!is_aligned(src, nbytes)) {
        if (posix_memalign(&staging, 4096, aligned_size) != 0) {
            perror("[llama-ssd] posix_memalign (write)");
            return false;
        }
        memcpy(staging, src, nbytes);
        // zero-pad the tail so that O_DIRECT is satisfied
        if (aligned_size > nbytes) {
            memset((char*)staging + nbytes, 0, aligned_size - nbytes);
        }
        buf = staging;
    }

    ssize_t written = pwrite(fd, buf, aligned_size, offset);

    if (staging) free(staging);

    if (written != (ssize_t)aligned_size) {
        if (written < 0) perror("[llama-ssd] pwrite");
        else fprintf(stderr, "[llama-ssd] pwrite: partial write (%zd / %zu bytes)\n", written, aligned_size);
        return false;
    }

    stats.write_bytes.fetch_add((uint64_t)nbytes,  std::memory_order_relaxed);
    stats.write_ops  .fetch_add(1,                 std::memory_order_relaxed);
    return true;
}

bool llama_ssd_kv_file::read_kv(void * dst, size_t nbytes, off_t offset) {
    if (fd < 0) return false;
    if (nbytes == 0) return true;

    void * buf  = dst;
    void * staging = nullptr;
    size_t aligned_size = (nbytes + 4095) & ~(size_t)4095;

    if (!is_aligned(dst, nbytes)) {
        if (posix_memalign(&staging, 4096, aligned_size) != 0) {
            perror("[llama-ssd] posix_memalign (read)");
            return false;
        }
        buf = staging;
    }

    ssize_t got = pread(fd, buf, aligned_size, offset);

    if (staging) {
        if (got >= 0) {
            memcpy(dst, staging, nbytes);
        }
        free(staging);
    }

    if (got < (ssize_t)nbytes) {
        if (got < 0) perror("[llama-ssd] pread");
        else fprintf(stderr, "[llama-ssd] pread: partial read (%zd / %zu bytes)\n", got, nbytes);
        return false;
    }

    stats.read_bytes.fetch_add((uint64_t)nbytes, std::memory_order_relaxed);
    stats.read_ops  .fetch_add(1,                std::memory_order_relaxed);
    return true;
}

bool llama_ssd_kv_file::read_blob(void * dst, size_t nbytes, off_t offset) {
    // OPT5: single pread for the entire sequence blob.
    // Intentionally delegates to read_kv — the distinction is semantic (one call
    // for the whole blob vs. per-layer slices), not structural at this layer.
    return read_kv(dst, nbytes, offset);
}
