// Hashcats GPU miner core (CUDA, keccak-256).
// workHash = keccak256(abi.encodePacked(address miner, uint256 nonce, uint256 prev, bytes32 anchor))
//   = 20 + 32 + 32 + 32 = 116 bytes, one keccak block (rate 136).
// Protocol (stdin/stdout, line based):
//   JOB <miner_hex40> <prev_hex64> <anchor_hex64> <target_hex64>
//   -> prints:  RATE <hashes_per_sec>
//               FOUND <nonce_decimal> <hash_hex64>
// Build: nvcc -O3 -arch=native -o hashcats_cuda hashcats_cuda.cu   (or -arch=sm_120 for RTX 5090)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <chrono>
#include <random>
#include <sys/select.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

__constant__ uint64_t c_tpl[17];      // template lanes (nonce bytes zeroed)
__constant__ uint64_t c_target[4];    // target as 4 big-endian 64-bit words (word0 = most significant)

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }
__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    return ((x & 0x00000000000000FFULL) << 56) | ((x & 0x000000000000FF00ULL) << 40) |
           ((x & 0x0000000000FF0000ULL) << 24) | ((x & 0x00000000FF000000ULL) << 8)  |
           ((x & 0x000000FF00000000ULL) >> 8)  | ((x & 0x0000FF0000000000ULL) >> 24) |
           ((x & 0x00FF000000000000ULL) >> 40) | ((x & 0xFF00000000000000ULL) >> 56);
}

__constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL };

__device__ __forceinline__ void keccakf(uint64_t s[25]) {
    uint64_t t, bc[5];
#pragma unroll 1
    for (int r = 0; r < 24; r++) {
        // theta
        for (int i = 0; i < 5; i++) bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) s[j + i] ^= t;
        }
        // rho + pi
        t = s[1];
        s[1]  = rotl64(s[6], 44);  s[6]  = rotl64(s[9], 20);  s[9]  = rotl64(s[22], 61); s[22] = rotl64(s[14], 39);
        s[14] = rotl64(s[20], 18); s[20] = rotl64(s[2], 62);  s[2]  = rotl64(s[12], 43); s[12] = rotl64(s[13], 25);
        s[13] = rotl64(s[19], 8);  s[19] = rotl64(s[23], 56); s[23] = rotl64(s[15], 41); s[15] = rotl64(s[4], 27);
        s[4]  = rotl64(s[24], 14); s[24] = rotl64(s[21], 2);  s[21] = rotl64(s[8], 55);  s[8]  = rotl64(s[16], 45);
        s[16] = rotl64(s[5], 36);  s[5]  = rotl64(s[3], 28);  s[3]  = rotl64(s[18], 21); s[18] = rotl64(s[17], 15);
        s[17] = rotl64(s[11], 10); s[11] = rotl64(s[7], 6);   s[7]  = rotl64(s[10], 3);  s[10] = rotl64(t, 1);
        // chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) bc[i] = s[j + i];
            for (int i = 0; i < 5; i++) s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        // iota
        s[0] ^= RC[r];
    }
}

// nonce occupies bytes 44..51 (big-endian uint64 inside the uint256 nonce word, bytes 20..51).
// lane 5 = bytes 40..47 (little-endian), lane 6 = bytes 48..55.
__global__ void mine_kernel(uint64_t base, uint32_t per_thread, uint64_t *out_nonce, uint64_t *out_hash, unsigned int *out_cnt) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = base + tid * per_thread;
    for (uint32_t k = 0; k < per_thread; k++, nonce++) {
        uint64_t s[25];
#pragma unroll
        for (int i = 0; i < 17; i++) s[i] = c_tpl[i];
#pragma unroll
        for (int i = 17; i < 25; i++) s[i] = 0;
        // big-endian nonce bytes n0..n7 -> positions 44..51
        uint64_t be = bswap64(nonce);            // now byte0 of 'be' (LSB) = n0
        s[5] |= (be & 0xFFFFFFFFULL) << 32;      // n0..n3 -> bytes 44..47 (upper half of lane 5)
        s[6] |= (be >> 32) & 0xFFFFFFFFULL;      // n4..n7 -> bytes 48..51 (lower half of lane 6)
        keccakf(s);
        uint64_t h0 = bswap64(s[0]);
        if (h0 > c_target[0]) continue;
        if (h0 == c_target[0]) {
            uint64_t h1 = bswap64(s[1]);
            if (h1 > c_target[1]) continue;
            if (h1 == c_target[1]) {
                uint64_t h2 = bswap64(s[2]);
                if (h2 > c_target[2]) continue;
                if (h2 == c_target[2] && bswap64(s[3]) > c_target[3]) continue;
            }
        }
        unsigned int idx = atomicAdd(out_cnt, 1u);
        if (idx < 8) {
            out_nonce[idx] = nonce;
            out_hash[idx * 4 + 0] = h0;
            out_hash[idx * 4 + 1] = bswap64(s[1]);
            out_hash[idx * 4 + 2] = bswap64(s[2]);
            out_hash[idx * 4 + 3] = bswap64(s[3]);
        }
    }
}

static int hexval(char c) { if (c >= '0' && c <= '9') return c - '0'; c |= 0x20; if (c >= 'a' && c <= 'f') return c - 'a' + 10; return -1; }
static bool hex2bytes(const std::string &h, uint8_t *out, size_t n) {
    std::string s = h; if (s.rfind("0x", 0) == 0) s = s.substr(2);
    if (s.size() != n * 2) return false;
    for (size_t i = 0; i < n; i++) { int a = hexval(s[2*i]), b = hexval(s[2*i+1]); if (a < 0 || b < 0) return false; out[i] = (uint8_t)(a * 16 + b); }
    return true;
}

static bool build_job(const std::string &miner, const std::string &prev, const std::string &anchor, const std::string &target) {
    uint8_t buf[136]; memset(buf, 0, sizeof buf);
    if (!hex2bytes(miner, buf, 20) || !hex2bytes(prev, buf + 52, 32) || !hex2bytes(anchor, buf + 84, 32)) return false;
    buf[116] = 0x01; buf[135] = 0x80;               // keccak padding (0x01 ... 0x80)
    uint64_t tpl[17];
    for (int i = 0; i < 17; i++) { uint64_t v = 0; for (int b = 0; b < 8; b++) v |= (uint64_t)buf[i*8+b] << (8*b); tpl[i] = v; }
    uint8_t tb[32]; if (!hex2bytes(target, tb, 32)) return false;
    uint64_t tw[4];
    for (int i = 0; i < 4; i++) { uint64_t v = 0; for (int b = 0; b < 8; b++) v = (v << 8) | tb[i*8+b]; tw[i] = v; }
    CK(cudaMemcpyToSymbol(c_tpl, tpl, sizeof tpl));
    CK(cudaMemcpyToSymbol(c_target, tw, sizeof tw));
    return true;
}

static bool stdin_ready() {
    fd_set fds; FD_ZERO(&fds); FD_SET(0, &fds);
    struct timeval tv = {0, 0};
    return select(1, &fds, nullptr, nullptr, &tv) > 0;
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    printf("INFO device %s sm=%d.%d SMs=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);

    uint64_t *d_nonce, *d_hash; unsigned int *d_cnt;
    CK(cudaMalloc(&d_nonce, 8 * sizeof(uint64_t)));
    CK(cudaMalloc(&d_hash, 32 * sizeof(uint64_t)));
    CK(cudaMalloc(&d_cnt, sizeof(unsigned int)));

    const int threads = 256;
    const int blocks = p.multiProcessorCount * 64;
    const uint32_t per_thread = 64;
    const uint64_t per_launch = (uint64_t)blocks * threads * per_thread;

    std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count() ^ getpid());
    uint64_t base = rng();
    bool have_job = false;
    std::string line;
    uint64_t hashes = 0; auto t0 = std::chrono::steady_clock::now();

    while (true) {
        while (stdin_ready() || !have_job) {
            char lb[1024];
            if (!fgets(lb, sizeof lb, stdin)) { printf("INFO stdin closed, exiting\n"); return 0; }
            line = lb; while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            char cmd[16], a[128], b[128], c[128], d[128];
            if (sscanf(line.c_str(), "%15s %127s %127s %127s %127s", cmd, a, b, c, d) == 5 && strcmp(cmd, "JOB") == 0) {
                if (build_job(a, b, c, d)) { have_job = true; base = rng(); printf("INFO job accepted\n"); }
                else printf("ERR bad job\n");
            } else if (strcmp(line.c_str(), "STOP") == 0) { have_job = false; printf("INFO stopped\n"); }
            else if (!line.empty()) printf("ERR unknown line\n");
            if (!have_job) continue;
        }
        CK(cudaMemset(d_cnt, 0, sizeof(unsigned int)));
        mine_kernel<<<blocks, threads>>>(base, per_thread, d_nonce, d_hash, d_cnt);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        unsigned int cnt = 0; CK(cudaMemcpy(&cnt, d_cnt, sizeof cnt, cudaMemcpyDeviceToHost));
        if (cnt > 0) {
            uint64_t hn[8], hh[32];
            CK(cudaMemcpy(hn, d_nonce, sizeof hn, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hh, d_hash, sizeof hh, cudaMemcpyDeviceToHost));
            for (unsigned int i = 0; i < cnt && i < 8; i++)
                printf("FOUND %llu %016llx%016llx%016llx%016llx\n", (unsigned long long)hn[i],
                       (unsigned long long)hh[i*4], (unsigned long long)hh[i*4+1], (unsigned long long)hh[i*4+2], (unsigned long long)hh[i*4+3]);
        }
        base += per_launch; hashes += per_launch;
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        if (dt >= 2.0) { printf("RATE %.0f\n", hashes / dt); hashes = 0; t0 = t1; }
    }
}
