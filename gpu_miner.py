"""
Hashcats native GPU miner — оркестратор (v2).
Запускает ./hashcats_cuda (CUDA keccak), следит за состоянием контракта,
подставляет актуальные prev/anchor/target и отправляет найденное решение
транзакцией mine(nonce, anchorBlock) с приложенной ценой.

Отличия v2: один batch-запрос к RPC вместо четырёх, несколько RPC с ротацией,
экспоненциальная пауза на 429/ошибки, повторы при старте, автоперезапуск CUDA.

ENV:
    MINER_PRIVATE_KEY   0x... приватный ключ burner-кошелька (обязательно)
    RPC_URLS            список RPC через запятую (по умолчанию 8 публичных узлов)
    CUDA_BIN            путь к бинарнику (по умолчанию ./hashcats_cuda)
    REFRESH_SEC         период опроса состояния (по умолчанию 2.5 с; anchor живёт ~25 с)
"""
import os
import random
import subprocess
import sys
import threading
import time
from queue import Queue, Empty

import requests
from eth_account import Account
from web3 import Web3

PRIVATE_KEY = os.environ.get("MINER_PRIVATE_KEY", "").strip()
DEFAULT_RPCS = ",".join([
    "https://robinhood-rpc.publicnode.com",
    "https://rpc.nodeflare.app/robinhood/public",
    "https://rpc-robinhood.blockmachine.io",
    "https://robinhood.drpc.org",
    "https://robinhood.api.pocket.network",
    "https://rpc.mainnet.chain.robinhood.com",
    "https://robinhood.rpc.blxrbdn.com",
    "https://rpc.ordofi.network",
])
RPC_URLS = [u.strip() for u in os.environ.get("RPC_URLS", DEFAULT_RPCS).split(",") if u.strip()]
random.shuffle(RPC_URLS)   # каждая машина начинает с другого узла — нагрузка размазывается
CUDA_BIN    = os.environ.get("CUDA_BIN", "./hashcats_cuda")
REFRESH_SEC = float(os.environ.get("REFRESH_SEC", "0.8"))
CHAIN_ID    = 4663
CONTRACT    = "0xCA75DF55Cc9C476DB27a7375D1fc8E794cf80721"

# селекторы (проверены против контракта)
SEL_CURRENT_ANCHOR = "0xcd809b11"
SEL_PREV_WORK      = "0xa4da5da2"
SEL_TARGET_FOR     = "0x16ccc8c0"
SEL_MINT_PRICE     = "0x6817c76c"

ABI = [
    {"type":"function","name":"mine","inputs":[{"name":"nonce","type":"uint256"},{"name":"anchorBlock","type":"uint256"}],"outputs":[{"name":"tokenId","type":"uint256"}],"stateMutability":"payable"},
]

if not (PRIVATE_KEY.startswith("0x") and len(PRIVATE_KEY) == 66):
    print("MINER_PRIVATE_KEY не задан или неверный (нужно 0x + 64 hex)"); sys.exit(1)

acct = Account.from_key(PRIVATE_KEY)
ADDR = acct.address


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------- RPC с ротацией и backoff ----------------
class Rpc:
    def __init__(self, urls):
        self.urls = urls
        self.i = random.randrange(len(urls))
        self.backoff = 0.0
        self.sess = requests.Session()

    @property
    def url(self):
        return self.urls[self.i]

    def rotate(self):
        self.i = (self.i + 1) % len(self.urls)

    def batch(self, calls, timeout=15):
        """calls: list of (method, params). Возвращает list результатов. Бросает при ошибке."""
        payload = [{"jsonrpc": "2.0", "id": k + 1, "method": m, "params": p} for k, (m, p) in enumerate(calls)]
        r = self.sess.post(self.url, json=payload, timeout=timeout)
        if r.status_code == 429:
            raise RuntimeError("429 Too Many Requests")
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict):  # сервер не поддержал batch
            j = [j]
        by_id = {x.get("id"): x for x in j}
        out = []
        for k in range(len(calls)):
            x = by_id.get(k + 1)
            if not x or "error" in x:
                raise RuntimeError(f"rpc error: {x.get('error') if x else 'missing'}")
            out.append(x["result"])
        return out

    def call_with_retry(self, calls, attempts=10):
        last = None
        for n in range(attempts):
            try:
                res = self.batch(calls)
                self.backoff = max(0.0, self.backoff / 2)
                return res
            except Exception as e:
                last = e
                if "429" in str(e):
                    self.backoff = min(20.0, (self.backoff or 1.0) * 2)
                else:
                    self.backoff = min(5.0, (self.backoff or 0.5) * 1.5)
                wait = self.backoff * (0.5 + random.random())
                log(f"[RPC] {self.url.split('//')[1][:32]} -> {str(e)[:50]}; пауза {wait:.1f}s, переключаюсь")
                self.rotate()
                time.sleep(wait)
        raise RuntimeError(f"RPC недоступен: {last}")


rpc = Rpc(RPC_URLS)


def w3_for(url):
    return Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 30}))


def bits_of(target: int) -> int:
    return 256 - target.bit_length()


def local_hash(nonce: int, prev: int, anchor: bytes) -> int:
    data = bytes.fromhex(ADDR[2:]) + nonce.to_bytes(32, "big") + prev.to_bytes(32, "big") + anchor
    return int.from_bytes(Web3.keccak(data), "big")


def read_state():
    """Одним batch-запросом: currentAnchor, prevWork, targetFor(me), mintPrice."""
    to = CONTRACT
    addr_word = ADDR[2:].lower().rjust(64, "0")
    res = rpc.call_with_retry([
        ("eth_call", [{"to": to, "data": SEL_CURRENT_ANCHOR}, "latest"]),
        ("eth_call", [{"to": to, "data": SEL_PREV_WORK}, "latest"]),
        ("eth_call", [{"to": to, "data": SEL_TARGET_FOR + addr_word}, "latest"]),
        ("eth_call", [{"to": to, "data": SEL_MINT_PRICE}, "latest"]),
        ("eth_gasPrice", []),
    ])
    ca = res[0][2:]
    anchor_block = int(ca[:64], 16)
    anchor = bytes.fromhex(ca[64:128])
    prev = int(res[1], 16)
    target = int(res[2], 16)
    price = int(res[3], 16)
    gas_price = int(res[4], 16)
    return {"anchor_block": anchor_block, "anchor": anchor, "prev": prev, "target": target,
            "price": price, "gas_price": gas_price}


def get_balance():
    res = rpc.call_with_retry([("eth_getBalance", [ADDR, "latest"])])
    return int(res[0], 16)


def quick_prev():
    """Быстрое чтение только prevWork с самого быстрого ответившего RPC (для проверки перед отправкой)."""
    import concurrent.futures as _cf
    def one(u):
        r = requests.post(u, json={"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                                   "params": [{"to": CONTRACT, "data": SEL_PREV_WORK}, "latest"]}, timeout=4)
        return int(r.json()["result"], 16)
    with _cf.ThreadPoolExecutor(max_workers=len(RPC_URLS)) as ex:
        futs = [ex.submit(one, u) for u in RPC_URLS]
        for f in _cf.as_completed(futs):
            try:
                return f.result()
            except Exception:
                continue
    return None


TX_NONCE = None   # кэш nonce кошелька, чтобы не ходить в RPC в критический момент
GAS_LIMIT = 700_000


def refresh_tx_nonce():
    global TX_NONCE
    res = rpc.call_with_retry([("eth_getTransactionCount", [ADDR, "pending"])])
    TX_NONCE = int(res[0], 16)
    return TX_NONCE


def _broadcast(raw_hex):
    """Рассылает подписанную транзакцию на ВСЕ RPC разом — включит тот, кто быстрее.
    Возвращает (hash, отправивший_url) или бросает, если никто не принял."""
    import concurrent.futures as _cf
    results, errs = [], []

    def send(u):
        r = requests.post(u, json={"jsonrpc": "2.0", "id": 1, "method": "eth_sendRawTransaction",
                                   "params": [raw_hex]}, timeout=8)
        j = r.json()
        if j.get("result"):
            return (j["result"], u)
        raise RuntimeError(str(j.get("error"))[:80])

    with _cf.ThreadPoolExecutor(max_workers=len(RPC_URLS)) as ex:
        futs = {ex.submit(send, u): u for u in RPC_URLS}
        for f in _cf.as_completed(futs):
            try:
                return f.result()
            except Exception as e:
                errs.append(str(e))
    # "already known" = транзакция уже принята другим RPC — это успех
    for e in errs:
        if "already known" in e.lower() or "known transaction" in e.lower():
            return ("0x" + Web3.keccak(hexstr=raw_hex).hex().replace("0x", ""), "already-known")
    raise RuntimeError("; ".join(errs)[:150])


def submit(nonce: int, anchor_block: int, st: dict):
    """Мгновенная отправка: без estimate_gas, фиксированный газ, рассылка на все RPC параллельно.
    Если решение протухло, транзакция откатится — value вернётся, сгорит только газ (~$0.01)."""
    global TX_NONCE
    if TX_NONCE is None:
        refresh_tx_nonce()
    w3 = w3_for(rpc.url)
    c = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT), abi=ABI)
    data = c.encode_abi("mine", args=[nonce, anchor_block]) if hasattr(c, "encode_abi") else \
        c.encodeABI(fn_name="mine", args=[nonce, anchor_block])
    for attempt in range(3):
        tx = {
            "to": Web3.to_checksum_address(CONTRACT), "from": ADDR, "value": st["price"], "data": data,
            "chainId": CHAIN_ID, "nonce": TX_NONCE, "gas": GAS_LIMIT,
            # На Arbitrum Orbit порядок в блоке НЕ зависит от газа (секвенсер сортирует по времени прихода),
            # поэтому priority=0 как у самых успешных майнеров — газ не тратится зря.
            "maxFeePerGas": max(int(st.get("gas_price", 0) * 2), Web3.to_wei(1, "gwei")),
            "maxPriorityFeePerGas": 0,
            "type": 2,
        }
        signed = acct.sign_transaction(tx)
        raw_hex = "0x" + signed.raw_transaction.hex().replace("0x", "")
        try:
            h, via = _broadcast(raw_hex)
            TX_NONCE += 1
            log(f"[TX] sent {h}  value={st['price']/1e18:.5f} ETH nonce={tx['nonce']} via {via.split('//')[-1][:24]}")
            rc = w3.eth.wait_for_transaction_receipt(h, timeout=180)
            log(f"[TX] status={rc['status']} block={rc['blockNumber']} gasUsed={rc['gasUsed']}")
            return rc["status"] == 1
        except Exception as e:
            msg = str(e).lower()
            if "nonce" in msg or "already known" in msg or "replacement" in msg:
                log(f"[TX] проблема с nonce ({str(e)[:60]}) — перечитываю и повторяю")
                refresh_tx_nonce(); continue
            raise
    raise RuntimeError("не удалось отправить: nonce конфликтует 3 раза подряд")


# ---------------- CUDA process ----------------
class Cuda:
    def __init__(self):
        self.proc = None
        self.q: Queue = Queue()
        self.start()

    def start(self):
        self.proc = subprocess.Popen([CUDA_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        log(f"[cuda] started pid={self.proc.pid}")

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def send(self, line):
        try:
            self.proc.stdin.write(line + "\n"); self.proc.stdin.flush()
        except Exception as e:
            log(f"[cuda] write failed: {e}")

    def restart(self):
        try:
            self.proc.kill()
        except Exception:
            pass
        time.sleep(2)
        self.start()


def main():
    log(f"[*] miner {ADDR}  rpc={RPC_URLS}")
    for n in range(20):
        try:
            bal = get_balance()
            log(f"[*] balance={bal/1e18:.5f} ETH")
            if bal < 0.021e18:
                log("[!] на кошельке меньше 0.021 ETH — найденное решение не отправится, пополните")
            break
        except Exception as e:
            log(f"[WARN] balance read failed ({str(e)[:60]}), повтор {n+1}/20")
            time.sleep(5)
    if not os.path.exists(CUDA_BIN):
        log(f"[!] нет бинарника {CUDA_BIN}"); sys.exit(1)

    for n in range(20):
        try:
            refresh_tx_nonce(); break
        except Exception as e:
            log(f"[WARN] nonce read failed ({str(e)[:60]}), повтор"); time.sleep(5)
    cuda = Cuda()
    state = None
    last_refresh = 0.0
    counters = {"mined": 0, "fails": 0, "stale": 0}
    clock = threading.Lock()

    def send_job(st):
        cuda.send(f"JOB {ADDR[2:]} {st['prev']:064x} {st['anchor'].hex()} {st['target']:064x}")

    def submit_worker(nonce, anchor_block, st):
        # Проверка перед отправкой: если prev уже сменился, решение мёртвое — не шлём (экономим газ).
        try:
            fresh = quick_prev()
            if fresh is not None and fresh != st["prev"]:
                with clock:
                    counters["stale"] += 1
                log("[skip] prev сменился до отправки — решение протухло, не шлю")
                return
        except Exception:
            pass
        try:
            ok = submit(nonce, anchor_block, st)
        except Exception as e:
            ok = False
            log(f"[FAIL] submit: {str(e)[:200]}")
        with clock:
            if ok:
                counters["mined"] += 1
                log(f"[OK] КОТ ДОБЫТ! всего: {counters['mined']}")
            else:
                counters["fails"] += 1

    ANCHOR_REFRESH_BLOCKS = 120   # обновлять якорь заданию, только когда он состарился на 120 блоков (окно 250)
    while True:
        now = time.time()
        if now - last_refresh >= REFRESH_SEC:
            last_refresh = now
            try:
                st = read_state()
                # Якорь меняется каждый блок — НЕ пересобираем задание из-за него каждый раз.
                # Пересобираем только при смене prev/target, или когда якорь задания устарел.
                anchor_stale = (state is not None and
                                st["anchor_block"] - state["anchor_block"] > ANCHOR_REFRESH_BLOCKS)
                prev_changed = state is None or st["prev"] != state["prev"]
                target_changed = state is not None and st["target"] != state["target"]
                if prev_changed or target_changed or anchor_stale:
                    if prev_changed:
                        log(f"[STATE] prev changed | bits={bits_of(st['target'])} "
                            f"price={st['price']/1e18:.5f} ETH anchorBlock={st['anchor_block']}")
                    state = st
                    send_job(st)
                else:
                    # prev тот же — обновим в state только цену/якорь для справки, задание не трогаем
                    state["price"] = st["price"]; state["gas_price"] = st["gas_price"]
            except Exception as e:
                log(f"[WARN] state read failed: {str(e)[:100]}")

        try:
            line = cuda.q.get(timeout=0.3)
        except Empty:
            continue
        if line is None:
            log("[!] CUDA процесс завершился — перезапускаю")
            cuda.restart()
            if state:
                send_job(state)
            continue
        if line.startswith("RATE"):
            rate = float(line.split()[1])
            b = bits_of(state["target"]) if state else 0
            eta = (2 ** b) / rate / 3600 if rate and b else 0
            log(f"[RATE] {rate/1e9:.2f} GH/s | target {b or '?'} bits | ожидание ~{eta:.1f} ч | mined={counters['mined']} fails={counters['fails']} stale={counters['stale']}")
        elif line.startswith("FOUND"):
            _, nonce_s, hash_hex = line.split()
            nonce = int(nonce_s)
            st = state
            if not st:
                continue
            lh = local_hash(nonce, st["prev"], st["anchor"])
            if f"{lh:064x}" != hash_hex:
                log(f"[!] GPU/CPU hash mismatch — GPU врёт, пропускаю"); continue
            if lh >= st["target"]:
                log("[!] хэш не ниже текущей цели — пропускаю"); continue
            log(f"[FOUND] nonce={nonce} ({bits_of(lh)} zero bits) — отправляю (карта продолжает считать)")
            # Отправка в фоне, ядро НЕ останавливаем — карта майнит дальше, не простаивает на 0%.
            threading.Thread(target=submit_worker,
                             args=(nonce, st["anchor_block"], dict(st)), daemon=True).start()
        elif line.startswith("INFO") or line.startswith("ERR") or "error" in line.lower():
            log("[cuda]", line)


if __name__ == "__main__":
    main()
