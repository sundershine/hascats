// Hashcats GPU miner core (CUDA, keccak-256).
// workHash = keccak256(abi.encodePacked(address miner, uint256 nonce, uint256 prev, bytes32 anchor))
//   = 20 + 32 + 32 + 32 = 116 bytes, one keccak block (rate 136).
// Protocol (stdin/stdout, line based):
//   JOB <miner_hex40> <prev_hex64> <anchor_hex64> <target_hex64> [anchorBlock]
//   TX  <txNonce> <valueWei> <maxFeeWei>          (параметры транзакции, обновляются каждый опрос)
//   -> prints:  RATE <hashes_per_sec>
//               FOUND <nonce_decimal> <hash_hex64>
//               SENT <txhash> <nonce> <txNonce>     (native tx-mode: транза уже ушла в сеть)
//               TXRES ok|err <ms> <url> [msg]
// Native tx-mode: если задан env MINER_PRIVATE_KEY, найденное решение подписывается и рассылается
// на все RPC (env RPC_URLS, через запятую) ПРЯМО ОТСЮДА, минуя пайп в Python (минус ~1-3 мс).
// Без ключа — старое поведение (FOUND печатается, отправляет Python).
// Build: nvcc -O3 -arch=native -o hashcats_cuda hashcats_cuda.cu -lsecp256k1 -lcurl -lpthread
//   без библиотек (старый режим): nvcc -O3 -arch=native -DHC_NO_NATIVE -o hashcats_cuda hashcats_cuda.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <mutex>
#include <thread>
#include <atomic>
#include <memory>
#include <cstdarg>
#include <cctype>
#include <sys/select.h>
#include <unistd.h>
#include <cuda_runtime.h>

static std::mutex g_out;                 // printf из разных потоков
static void out(const char *fmt, ...) {
    std::lock_guard<std::mutex> lk(g_out);
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
}
static bool g_native = false;

#ifdef HC_NO_NATIVE
// Сборка без libsecp256k1/libcurl (fallback): транзакции шлёт Python.
static void init_native() { out("INFO txmode python (собрано с HC_NO_NATIVE)\n"); }
static void native_submit(uint64_t) {}
static std::mutex g_txm; static uint64_t g_anchor_block = 0, g_py_nonce = 0, g_next_nonce = 0; static bool g_tx_ready = false;
struct { std::string value_wei, max_fee_wei; } g_tx;
#else
#include "hc_tx.h"

// ---------------- native tx submit ----------------
static hc::Wallet g_wallet;
static std::vector<std::string> g_rpcs;
static std::mutex g_txm;
static hc::TxParams g_tx;                // защищено g_txm
static bool g_tx_ready = false;
static uint64_t g_anchor_block = 0;      // из JOB (5-й аргумент) — якорь текущего задания
static uint64_t g_py_nonce = 0;          // tx-nonce, который прислал Python (pending count)
static uint64_t g_next_nonce = 0;        // наш локальный счётчик (после каждой отправки +1)

static void split_csv(const std::string &s, std::vector<std::string> &v) {
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(',', a); if (b == std::string::npos) b = s.size();
        std::string t = s.substr(a, b - a);
        while (!t.empty() && isspace((unsigned char)t.front())) t.erase(t.begin());
        while (!t.empty() && isspace((unsigned char)t.back())) t.pop_back();
        if (!t.empty()) v.push_back(t);
        a = b + 1;
    }
}

static void init_native() {
    const char *k = getenv("MINER_PRIVATE_KEY");
    if (!k || !*k) { out("INFO txmode python (MINER_PRIVATE_KEY не задан)\n"); return; }
    if (!g_wallet.init(k)) { out("INFO txmode python (плохой ключ)\n"); return; }
    const char *r = getenv("RPC_URLS");
    split_csv(r ? r : "", g_rpcs);
    if (g_rpcs.empty()) split_csv(
        "https://robinhood-rpc.publicnode.com,https://rpc.nodeflare.app/robinhood/public,"
        "https://rpc-robinhood.blockmachine.io,https://robinhood.drpc.org,https://robinhood.api.pocket.network,"
        "https://rpc.mainnet.chain.robinhood.com,https://robinhood.rpc.blxrbdn.com,https://rpc.ordofi.network", g_rpcs);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    hc::unhex("0xCA75DF55Cc9C476DB27a7375D1fc8E794cf80721", g_tx.to);
    g_native = true;
    out("INFO txmode native address %s rpcs=%zu\n", hc::hex(g_wallet.address).c_str(), g_rpcs.size());
}

// Рассылает и ждёт все ответы; возвращает true если хоть один RPC принял. nonce_err=true если все упали по nonce.
static bool send_once(const hc::TxParams &p, uint64_t nonce, uint64_t anchor_block, bool &nonce_err) {
    hc::bytes raw, h;
    if (!hc::build_mine_tx(g_wallet, p, nonce, anchor_block, raw, h)) { out("ERR sign failed\n"); return false; }
    std::string raw_hex = hc::hex(raw);
    out("SENT %s %llu %llu\n", hc::hex(h).c_str(), (unsigned long long)nonce, (unsigned long long)p.tx_nonce);
    auto ok = std::make_shared<std::atomic<int>>(0), err = std::make_shared<std::atomic<int>>(0), nerr = std::make_shared<std::atomic<int>>(0);
    auto done = std::make_shared<std::atomic<int>>(0);
    size_t n = g_rpcs.size();
    hc::broadcast(g_rpcs, raw_hex, [=](const hc::SendResult &r) {
        std::string u = r.url; size_t s = u.find("//"); if (s != std::string::npos) u = u.substr(s + 2); if (u.size() > 28) u.resize(28);
        if (r.ok) { (*ok)++; out("TXRES ok %.0f %s %s\n", r.ms, u.c_str(), r.msg.c_str()); }
        else {
            std::string low = r.msg; for (auto &c : low) c = (char)tolower(c);
            if (low.find("nonce") != std::string::npos) (*nerr)++;
            (*err)++; out("TXRES err %.0f %s %s\n", r.ms, u.c_str(), r.msg.c_str());
        }
        (*done)++;
    });
    // ждём первый ok или все ответы (не дольше 9 с)
    auto t0 = std::chrono::steady_clock::now();
    while (*ok == 0 && (size_t)*done < n && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(9))
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    nonce_err = (*ok == 0 && *nerr > 0 && (size_t)(*nerr) >= (size_t)*done / 2 + 1);
    return *ok > 0;
}

static void native_submit(uint64_t nonce) {
    hc::TxParams p; uint64_t anchor_block, pyn;
    {
        std::lock_guard<std::mutex> lk(g_txm);
        if (!g_tx_ready) { out("ERR tx params not set yet — FOUND not sent\n"); return; }
        p = g_tx; anchor_block = g_anchor_block; pyn = g_py_nonce;
        if (g_next_nonce < pyn) g_next_nonce = pyn;
        p.tx_nonce = g_next_nonce;
        g_next_nonce++;                 // резервируем сразу: следующий FOUND возьмёт +1
    }
    bool nonce_err = false;
    if (send_once(p, nonce, anchor_block, nonce_err)) return;
    if (nonce_err && p.tx_nonce != pyn) {   // рассинхрон — повтор с nonce от Python
        out("INFO nonce resync %llu -> %llu\n", (unsigned long long)p.tx_nonce, (unsigned long long)pyn);
        { std::lock_guard<std::mutex> lk(g_txm); g_next_nonce = pyn + 1; }
        p.tx_nonce = pyn;
        bool dummy; send_once(p, nonce, anchor_block, dummy);
    }
}
#endif // HC_NO_NATIVE

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
                if (h2 == c_target[2] && bswap64(s[3]) >= c_target[3]) continue;
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
    init_native();

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
            char cmd[16], a[128], b[128], c[128], d[128], e[128];
            int nf = sscanf(line.c_str(), "%15s %127s %127s %127s %127s %127s", cmd, a, b, c, d, e);
            if (nf >= 5 && strcmp(cmd, "JOB") == 0) {
                if (build_job(a, b, c, d)) {
                    have_job = true; base = rng();
                    if (nf >= 6) { std::lock_guard<std::mutex> lk(g_txm); g_anchor_block = strtoull(e, nullptr, 10); }
                    out("INFO job accepted\n");
                } else out("ERR bad job\n");
            } else if (nf == 4 && strcmp(cmd, "TX") == 0) {
                // TX <txNonce> <valueWei> <maxFeeWei>
                std::lock_guard<std::mutex> lk(g_txm);
                g_py_nonce = strtoull(a, nullptr, 10);
                g_tx.value_wei = b; g_tx.max_fee_wei = c;
                if (!g_tx_ready) { g_next_nonce = g_py_nonce; g_tx_ready = true; out("INFO tx params ready nonce=%llu\n", (unsigned long long)g_py_nonce); }
            } else if (strcmp(line.c_str(), "STOP") == 0) { have_job = false; out("INFO stopped\n"); }
            else if (!line.empty()) out("ERR unknown line\n");
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
            for (unsigned int i = 0; i < cnt && i < 8; i++) {
                // Сначала — в сеть (подпись ~0.1 мс + POST на все RPC в фоне), потом уже лог. Ядро не ждёт.
                if (g_native) { uint64_t nn = hn[i]; std::thread([nn]() { native_submit(nn); }).detach(); }
                out("FOUND %llu %016llx%016llx%016llx%016llx\n", (unsigned long long)hn[i],
                    (unsigned long long)hh[i*4], (unsigned long long)hh[i*4+1], (unsigned long long)hh[i*4+2], (unsigned long long)hh[i*4+3]);
            }
        }
        base += per_launch; hashes += per_launch;
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        if (dt >= 2.0) { out("RATE %.0f\n", hashes / dt); hashes = 0; t0 = t1; }
    }
}
