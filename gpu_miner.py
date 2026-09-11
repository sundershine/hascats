"""
Hashcats native GPU miner — оркестратор.
Запускает ./hashcats_cuda (CUDA keccak), следит за состоянием контракта,
подставляет актуальные prev/anchor/target и отправляет найденное решение
транзакцией mine(nonce, anchorBlock) с приложенной ценой.

ENV:
    MINER_PRIVATE_KEY   0x... приватный ключ burner-кошелька (обязательно)
    RPC_URL             (по умолчанию https://rpc.mainnet.chain.robinhood.com)
    CUDA_BIN            путь к бинарнику (по умолчанию ./hashcats_cuda)
    REFRESH_SEC         как часто перечитывать anchor (по умолчанию 8 с; anchor живёт ~25 с)
    NONCE_MAX_BITS      64 (ядро перебирает 64-битный nonce)

Формат работы проверен против контракта:
    workHash = keccak256(abi.encodePacked(address miner, uint256 nonce, uint256 prev, bytes32 anchor))
    mine(uint256 nonce, uint256 anchorBlock) payable, value = mintPrice()
    target = targetFor(miner)   (учитывает личный и сетевой streak)
"""
import os
import subprocess
import sys
import threading
import time
from queue import Queue, Empty

from eth_account import Account
from web3 import Web3

PRIVATE_KEY = os.environ.get("MINER_PRIVATE_KEY", "").strip()
RPC_URL     = os.environ.get("RPC_URL", "https://rpc.mainnet.chain.robinhood.com")
CUDA_BIN    = os.environ.get("CUDA_BIN", "./hashcats_cuda")
REFRESH_SEC = float(os.environ.get("REFRESH_SEC", "8"))
CHAIN_ID    = 4663
CONTRACT    = "0xCA75DF55Cc9C476DB27a7375D1fc8E794cf80721"

ABI = [
    {"type":"function","name":"currentAnchor","inputs":[],"outputs":[{"name":"anchorBlock","type":"uint256"},{"name":"anchor","type":"bytes32"}],"stateMutability":"view"},
    {"type":"function","name":"prevWork","inputs":[],"outputs":[{"type":"uint256"}],"stateMutability":"view"},
    {"type":"function","name":"currentTarget","inputs":[],"outputs":[{"type":"uint256"}],"stateMutability":"view"},
    {"type":"function","name":"targetFor","inputs":[{"name":"miner","type":"address"}],"outputs":[{"type":"uint256"}],"stateMutability":"view"},
    {"type":"function","name":"mintPrice","inputs":[],"outputs":[{"type":"uint256"}],"stateMutability":"view"},
    {"type":"function","name":"currentBurst","inputs":[],"outputs":[{"type":"uint256"}],"stateMutability":"view"},
    {"type":"function","name":"personalBurst","inputs":[{"name":"miner","type":"address"}],"outputs":[{"type":"uint256"}],"stateMutability":"view"},
    {"type":"function","name":"totalMinted","inputs":[],"outputs":[{"type":"uint256"}],"stateMutability":"view"},
    {"type":"function","name":"workHash","inputs":[{"name":"miner","type":"address"},{"name":"nonce","type":"uint256"},{"name":"prev","type":"uint256"},{"name":"anchor","type":"bytes32"}],"outputs":[{"type":"uint256"}],"stateMutability":"pure"},
    {"type":"function","name":"mine","inputs":[{"name":"nonce","type":"uint256"},{"name":"anchorBlock","type":"uint256"}],"outputs":[{"name":"tokenId","type":"uint256"}],"stateMutability":"payable"},
]

if not (PRIVATE_KEY.startswith("0x") and len(PRIVATE_KEY) == 66):
    print("MINER_PRIVATE_KEY не задан или неверный (нужно 0x + 64 hex)"); sys.exit(1)

acct = Account.from_key(PRIVATE_KEY)
w3 = Web3(Web3.HTTPProvider(RPC_URL, request_kwargs={"timeout": 30}))
c = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT), abi=ABI)
ADDR = acct.address


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def bits_of(target: int) -> int:
    return 256 - target.bit_length()


def local_hash(nonce: int, prev: int, anchor: bytes) -> int:
    data = bytes.fromhex(ADDR[2:]) + nonce.to_bytes(32, "big") + prev.to_bytes(32, "big") + anchor
    return int.from_bytes(Web3.keccak(data), "big")


def read_state():
    anchor_block, anchor = c.functions.currentAnchor().call()
    prev = c.functions.prevWork().call()
    target = c.functions.targetFor(ADDR).call()
    price = c.functions.mintPrice().call()
    return {"anchor_block": anchor_block, "anchor": bytes(anchor), "prev": prev, "target": target, "price": price}


def submit(nonce: int, anchor_block: int, price: int):
    fn = c.functions.mine(nonce, anchor_block)
    base = {"from": ADDR, "value": price}
    gas = fn.estimate_gas(base)  # если решение уже протухло — здесь будет revert, транзакция не уйдёт
    tx = fn.build_transaction({
        **base,
        "chainId": CHAIN_ID,
        "nonce": w3.eth.get_transaction_count(ADDR, "pending"),
        "gas": int(gas * 1.3),
        "maxFeePerGas": int(w3.eth.gas_price * 3),
        "maxPriorityFeePerGas": 0,
        "type": 2,
    })
    signed = acct.sign_transaction(tx)
    h = w3.eth.send_raw_transaction(signed.raw_transaction)
    log(f"[TX] sent {h.hex()}  value={w3.from_wei(price,'ether')} ETH")
    rc = w3.eth.wait_for_transaction_receipt(h, timeout=180)
    log(f"[TX] status={rc['status']} block={rc['blockNumber']} gasUsed={rc['gasUsed']}")
    return rc["status"] == 1


def reader(proc, q):
    for line in proc.stdout:
        q.put(line.rstrip("\n"))
    q.put(None)


def main():
    bal = w3.eth.get_balance(ADDR)
    log(f"[*] miner {ADDR} balance={w3.from_wei(bal,'ether'):.5f} ETH  rpc={RPC_URL}")
    if not os.path.exists(CUDA_BIN):
        log(f"[!] нет бинарника {CUDA_BIN}. Соберите: nvcc -O3 -arch=native -o hashcats_cuda hashcats_cuda.cu"); sys.exit(1)

    proc = subprocess.Popen([CUDA_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    q: Queue = Queue()
    threading.Thread(target=reader, args=(proc, q), daemon=True).start()

    state = None
    last_refresh = 0.0
    mined = 0

    def send_job(st):
        line = f"JOB {ADDR[2:]} {st['prev']:064x} {st['anchor'].hex()} {st['target']:064x}\n"
        proc.stdin.write(line); proc.stdin.flush()

    while True:
        now = time.time()
        if now - last_refresh >= REFRESH_SEC:
            try:
                st = read_state()
                changed = (state is None or st["prev"] != state["prev"] or st["anchor"] != state["anchor"]
                           or st["target"] != state["target"])
                if changed:
                    if state is None or st["prev"] != state["prev"]:
                        log(f"[STATE] prev changed (кто-то заминтил) | bits={bits_of(st['target'])} "
                            f"price={w3.from_wei(st['price'],'ether')} ETH anchorBlock={st['anchor_block']}")
                    state = st
                    send_job(st)
                last_refresh = now
            except Exception as e:
                log(f"[WARN] state read failed: {e}")
                last_refresh = now - REFRESH_SEC + 2

        try:
            line = q.get(timeout=0.5)
        except Empty:
            continue
        if line is None:
            log("[!] CUDA процесс завершился"); sys.exit(2)
        if line.startswith("RATE"):
            rate = float(line.split()[1])
            need = 2 ** bits_of(state["target"]) if state else 0
            eta = need / rate / 3600 if rate else 0
            log(f"[RATE] {rate/1e9:.2f} GH/s | target {bits_of(state['target']) if state else '?'} bits | "
                f"ожидание ~{eta:.1f} ч | mined={mined}")
        elif line.startswith("FOUND"):
            _, nonce_s, hash_hex = line.split()
            nonce = int(nonce_s)
            st = state
            lh = local_hash(nonce, st["prev"], st["anchor"])
            if f"{lh:064x}" != hash_hex:
                log(f"[!] локальная проверка не совпала: gpu={hash_hex} cpu={lh:064x} — GPU врёт, пропускаю"); continue
            if lh > st["target"]:
                log("[!] хэш выше текущей цели (target ужесточился) — пропускаю"); continue
            log(f"[FOUND] nonce={nonce} hash={hash_hex} ({bits_of(lh)} zero bits) — отправляю")
            proc.stdin.write("STOP\n"); proc.stdin.flush()
            try:
                if submit(nonce, st["anchor_block"], st["price"]):
                    mined += 1
                    log(f"[OK] КОТ ДОБЫТ! всего: {mined}")
            except Exception as e:
                log(f"[FAIL] submit: {str(e)[:300]}")
            state = None; last_refresh = 0.0   # перечитать состояние и выдать новую задачу
        elif line.startswith("INFO") or line.startswith("ERR"):
            log("[cuda]", line)


if __name__ == "__main__":
    main()
