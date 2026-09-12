// hc_tx.h — сборка, подпись (EIP-1559) и рассылка транзакции mine(nonce, anchorBlock)
// прямо из CUDA-хост-кода, без пайпа в Python. Чистый C++17: libsecp256k1 + libcurl.
//   g++/nvcc ... -lsecp256k1 -lcurl -lpthread
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <secp256k1.h>
#include <secp256k1_recovery.h>
#include <curl/curl.h>

namespace hc {

typedef std::vector<uint8_t> bytes;

// ---------------- keccak-256 (host) ----------------
static const uint64_t KRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL };
static const int KROT[24] = {1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
static const int KPIL[24] = {10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};

static inline uint64_t rotl(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

static void keccakf_host(uint64_t s[25]) {
    uint64_t t, bc[5];
    for (int r = 0; r < 24; r++) {
        for (int i = 0; i < 5; i++) bc[i] = s[i] ^ s[i+5] ^ s[i+10] ^ s[i+15] ^ s[i+20];
        for (int i = 0; i < 5; i++) { t = bc[(i+4)%5] ^ rotl(bc[(i+1)%5], 1); for (int j = 0; j < 25; j += 5) s[j+i] ^= t; }
        t = s[1];
        for (int i = 0; i < 24; i++) { int j = KPIL[i]; bc[0] = s[j]; s[j] = rotl(t, KROT[i]); t = bc[0]; }
        for (int j = 0; j < 25; j += 5) { for (int i = 0; i < 5; i++) bc[i] = s[j+i]; for (int i = 0; i < 5; i++) s[j+i] ^= (~bc[(i+1)%5]) & bc[(i+2)%5]; }
        s[0] ^= KRC[r];
    }
}

static bytes keccak256(const uint8_t *data, size_t len) {
    uint64_t s[25]; memset(s, 0, sizeof s);
    const size_t rate = 136;
    uint8_t blk[136];
    size_t off = 0;
    while (true) {
        size_t n = len - off < rate ? len - off : rate;
        memset(blk, 0, rate); memcpy(blk, data + off, n);
        if (n < rate) { blk[n] ^= 0x01; blk[rate-1] ^= 0x80; }
        for (int i = 0; i < 17; i++) { uint64_t v = 0; for (int b = 0; b < 8; b++) v |= (uint64_t)blk[i*8+b] << (8*b); s[i] ^= v; }
        keccakf_host(s);
        off += n;
        if (n < rate) break;
        if (off == len) { // ровно кратно rate — нужен ещё блок с одним паддингом
            memset(blk, 0, rate); blk[0] ^= 0x01; blk[rate-1] ^= 0x80;
            for (int i = 0; i < 17; i++) { uint64_t v = 0; for (int b = 0; b < 8; b++) v |= (uint64_t)blk[i*8+b] << (8*b); s[i] ^= v; }
            keccakf_host(s); break;
        }
    }
    bytes out(32);
    for (int i = 0; i < 4; i++) for (int b = 0; b < 8; b++) out[i*8+b] = (uint8_t)(s[i] >> (8*b));
    return out;
}
static bytes keccak256(const bytes &v) { return keccak256(v.data(), v.size()); }

// ---------------- hex helpers ----------------
static std::string hex(const bytes &v, bool prefix = true) {
    static const char *d = "0123456789abcdef";
    std::string s = prefix ? "0x" : "";
    for (uint8_t b : v) { s += d[b >> 4]; s += d[b & 15]; }
    return s;
}
static int hv(char c) { if (c >= '0' && c <= '9') return c - '0'; c |= 0x20; if (c >= 'a' && c <= 'f') return c - 'a' + 10; return -1; }
static bool unhex(const std::string &h, bytes &out) {
    std::string s = h; if (s.rfind("0x", 0) == 0) s = s.substr(2);
    if (s.size() % 2) s = "0" + s;
    out.clear(); out.reserve(s.size() / 2);
    for (size_t i = 0; i < s.size(); i += 2) { int a = hv(s[i]), b = hv(s[i+1]); if (a < 0 || b < 0) return false; out.push_back((uint8_t)(a*16+b)); }
    return true;
}

// ---------------- 256-бит целые как big-endian bytes ----------------
// Десятичная строка -> big-endian bytes без ведущих нулей (для RLP) или 32 байта (для ABI).
static bytes dec_to_be(const std::string &dec) {
    bytes v; // little-endian накопитель
    for (char c : dec) {
        if (c < '0' || c > '9') continue;
        uint32_t carry = (uint32_t)(c - '0');
        for (size_t i = 0; i < v.size(); i++) { uint32_t x = v[i] * 10u + carry; v[i] = (uint8_t)x; carry = x >> 8; }
        while (carry) { v.push_back((uint8_t)carry); carry >>= 8; }
    }
    while (!v.empty() && v.back() == 0) v.pop_back();
    return bytes(v.rbegin(), v.rend());
}
static bytes u64_to_be(uint64_t x) { bytes v; while (x) { v.insert(v.begin(), (uint8_t)x); x >>= 8; } return v; }
static bytes pad32(const bytes &v) { bytes o(32, 0); if (v.size() <= 32) memcpy(o.data() + 32 - v.size(), v.data(), v.size()); return o; }

// ---------------- RLP ----------------
static bytes rlp_len(size_t n, uint8_t base) {
    bytes o;
    if (n < 56) { o.push_back((uint8_t)(base + n)); return o; }
    bytes l = u64_to_be(n); o.push_back((uint8_t)(base + 55 + l.size())); o.insert(o.end(), l.begin(), l.end()); return o;
}
static bytes rlp_bytes(const bytes &v) {
    if (v.size() == 1 && v[0] < 0x80) return bytes(1, v[0]);
    bytes o = rlp_len(v.size(), 0x80); o.insert(o.end(), v.begin(), v.end()); return o;
}
static bytes rlp_list(const std::vector<bytes> &items) {
    bytes body; for (auto &it : items) body.insert(body.end(), it.begin(), it.end());
    bytes o = rlp_len(body.size(), 0xc0); o.insert(o.end(), body.begin(), body.end()); return o;
}

// ---------------- кошелёк ----------------
struct Wallet {
    secp256k1_context *ctx = nullptr;
    uint8_t key[32];
    bytes address;   // 20 байт
    bool ok = false;

    bool init(const std::string &privhex) {
        bytes k; if (!unhex(privhex, k) || k.size() != 32) return false;
        memcpy(key, k.data(), 32);
        ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
        if (!secp256k1_ec_seckey_verify(ctx, key)) return false;
        secp256k1_pubkey pub;
        if (!secp256k1_ec_pubkey_create(ctx, &pub, key)) return false;
        uint8_t ser[65]; size_t sl = 65;
        secp256k1_ec_pubkey_serialize(ctx, ser, &sl, &pub, SECP256K1_EC_UNCOMPRESSED);
        bytes h = keccak256(ser + 1, 64);
        address.assign(h.begin() + 12, h.end());
        ok = true; return true;
    }
    // Возвращает (r, s, v) для хэша.
    bool sign(const bytes &hash32, bytes &r, bytes &s, int &v) const {
        secp256k1_ecdsa_recoverable_signature sig;
        if (!secp256k1_ecdsa_sign_recoverable(ctx, &sig, hash32.data(), key, nullptr, nullptr)) return false;
        uint8_t compact[64]; int recid = 0;
        secp256k1_ecdsa_recoverable_signature_serialize_compact(ctx, compact, &recid, &sig);
        r.assign(compact, compact + 32); s.assign(compact + 32, compact + 64); v = recid;
        // RLP-целые без ведущих нулей
        while (!r.empty() && r[0] == 0) r.erase(r.begin());
        while (!s.empty() && s[0] == 0) s.erase(s.begin());
        return true;
    }
};

// ---------------- параметры транзакции ----------------
struct TxParams {
    uint64_t chain_id = 4663;
    bytes to;                 // 20 байт
    uint64_t tx_nonce = 0;
    std::string value_wei;    // десятичная строка (mintPrice)
    std::string max_fee_wei;  // десятичная строка
    uint64_t gas_limit = 700000;
};

// selector: keccak("mine(uint256,uint256)")[:4]
static bytes selector_mine() {
    const char *sig = "mine(uint256,uint256)";
    bytes h = keccak256((const uint8_t*)sig, strlen(sig));
    return bytes(h.begin(), h.begin() + 4);
}

// Собирает подписанную EIP-1559 (type 2) транзакцию. Возвращает raw bytes (с префиксом 0x02) и её хэш.
static bool build_mine_tx(const Wallet &w, const TxParams &p, uint64_t nonce, uint64_t anchor_block, bytes &raw, bytes &txhash) {
    bytes data = selector_mine();
    bytes n32 = pad32(u64_to_be(nonce)), a32 = pad32(u64_to_be(anchor_block));
    data.insert(data.end(), n32.begin(), n32.end());
    data.insert(data.end(), a32.begin(), a32.end());

    std::vector<bytes> f = {
        rlp_bytes(u64_to_be(p.chain_id)),
        rlp_bytes(u64_to_be(p.tx_nonce)),
        rlp_bytes(bytes()),                        // maxPriorityFeePerGas = 0 (секвенсер Orbit сортирует по времени)
        rlp_bytes(dec_to_be(p.max_fee_wei)),
        rlp_bytes(u64_to_be(p.gas_limit)),
        rlp_bytes(p.to),
        rlp_bytes(dec_to_be(p.value_wei)),
        rlp_bytes(data),
        rlp_list({}),                              // accessList = []
    };
    bytes unsigned_payload = rlp_list(f);
    bytes to_hash; to_hash.push_back(0x02); to_hash.insert(to_hash.end(), unsigned_payload.begin(), unsigned_payload.end());
    bytes h = keccak256(to_hash);
    bytes r, s; int v;
    if (!w.sign(h, r, s, v)) return false;
    f.push_back(rlp_bytes(v ? u64_to_be(1) : bytes()));
    f.push_back(rlp_bytes(r));
    f.push_back(rlp_bytes(s));
    bytes signed_payload = rlp_list(f);
    raw.clear(); raw.push_back(0x02); raw.insert(raw.end(), signed_payload.begin(), signed_payload.end());
    txhash = keccak256(raw);
    return true;
}

// ---------------- рассылка на все RPC параллельно ----------------
static size_t curl_sink(char *ptr, size_t sz, size_t nm, void *ud) { ((std::string*)ud)->append(ptr, sz*nm); return sz*nm; }

struct SendResult { std::string url; bool ok; std::string msg; double ms; };

// Шлёт raw tx на каждый URL в своём потоке. on_result вызывается по мере прихода ответов (из разных потоков).
template <class F>
static void broadcast(const std::vector<std::string> &urls, const std::string &raw_hex, F on_result) {
    std::vector<std::thread> th;
    for (const auto &u : urls) {
        th.emplace_back([u, raw_hex, on_result]() {
            auto t0 = std::chrono::steady_clock::now();
            std::string body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_sendRawTransaction\",\"params\":[\"" + raw_hex + "\"]}";
            std::string resp;
            CURL *c = curl_easy_init();
            struct curl_slist *hdr = curl_slist_append(nullptr, "Content-Type: application/json");
            curl_easy_setopt(c, CURLOPT_URL, u.c_str());
            curl_easy_setopt(c, CURLOPT_POSTFIELDS, body.c_str());
            curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdr);
            curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, curl_sink);
            curl_easy_setopt(c, CURLOPT_WRITEDATA, &resp);
            curl_easy_setopt(c, CURLOPT_TIMEOUT_MS, 8000L);
            curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, 3000L);
            curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
            CURLcode rc = curl_easy_perform(c);
            curl_slist_free_all(hdr); curl_easy_cleanup(c);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            SendResult r{u, false, "", ms};
            if (rc != CURLE_OK) r.msg = curl_easy_strerror(rc);
            else if (resp.find("\"result\"") != std::string::npos && resp.find("\"error\"") == std::string::npos) r.ok = true;
            else {
                size_t p = resp.find("\"message\"");
                r.msg = p == std::string::npos ? resp.substr(0, 120) : resp.substr(p, 120);
                std::string low = r.msg; for (auto &ch : low) ch = (char)tolower(ch);
                if (low.find("already known") != std::string::npos || low.find("known transaction") != std::string::npos) { r.ok = true; r.msg = "already known"; }
            }
            on_result(r);
        });
    }
    for (auto &t : th) t.detach();
}

} // namespace hc
