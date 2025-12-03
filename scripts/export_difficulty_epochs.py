#!/usr/bin/env python3
import os
import sys
import csv
import requests

# Which chain (for cookie path)?
CHAIN = os.getenv("BTC_CHAIN", "mainnet")
HOME = os.path.expanduser("~")

if CHAIN == "mainnet":
    COOKIE_PATH = f"{HOME}/.bitcoin/.cookie"
elif CHAIN in ("testnet", "testnet3"):
    COOKIE_PATH = f"{HOME}/.bitcoin/testnet3/.cookie"
elif CHAIN == "regtest":
    COOKIE_PATH = f"{HOME}/.bitcoin/regtest/.cookie"
else:
    raise ValueError("Unknown BTC_CHAIN value")

def read_cookie(path):
    with open(path, "r") as f:
        data = f.read().strip()
    user, pwd = data.split(":", 1)
    return user, pwd

RPC_HOST = os.getenv("BTC_RPC_HOST", "127.0.0.1")
RPC_PORT = int(os.getenv("BTC_RPC_PORT", "8332"))
RPC_URL = f"http://{RPC_HOST}:{RPC_PORT}"

OUTPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "difficulty_epochs.csv"

def rpc_call(method, params=None, auth=None):
    payload = {
        "jsonrpc": "1.0",
        "id": "epoch-diff-export",
        "method": method,
        "params": params or [],
    }
    r = requests.post(RPC_URL, json=payload, auth=auth)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    return data["result"]

def main():
    print(f"Using cookie: {COOKIE_PATH}")
    user, pwd = read_cookie(COOKIE_PATH)
    auth = (user, pwd)

    tip = rpc_call("getblockcount", auth=auth)
    print("Tip height:", tip)

    heights = list(range(0, tip + 1, 2016))
    print("Sampling", len(heights), "epochs")

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["height", "timestamp", "difficulty"])

        for h in heights:
            bh = rpc_call("getblockhash", [h], auth=auth)
            header = rpc_call("getblockheader", [bh], auth=auth)
            ts = header["time"]
            diff = header["difficulty"]
            w.writerow([h, ts, diff])

    print("Wrote", OUTPUT_CSV)

if __name__ == "__main__":
    main()

