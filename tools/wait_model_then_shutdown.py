"""等 Ollama 模型拉取完成后自动关机。

用法：
    python tools/wait_model_then_shutdown.py                  # 默认 qwen3vl:8b，完成后 120 秒关机
    python tools/wait_model_then_shutdown.py --no-shutdown    # 只等不关机（调试用）
    python tools/wait_model_then_shutdown.py --model qwen3vl:8b --shutdown-delay 60

行为：
  1. 每 15 秒轮询 `ollama list`，模型出现即视为完成；
  2. 监控 blobs 目录占用，连续 STALL_SEC 秒没有增长 → 认为 pull 中断，自动重跑
     `ollama pull`（Ollama 自带断点续传，不会从头下）；
  3. 完成后等待 shutdown-delay 秒（留出取消时间），执行 shutdown /s /t 0。
     取消关机：在任意终端运行  shutdown /a
"""

import argparse
import os
import subprocess
import sys
import time

OLLAMA = os.path.expandvars(
    r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
)
BLOBS = os.path.expanduser(r"~\.ollama\models\blobs")
POLL = 15
STALL_SEC = 150


def run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {cmd[:1]} 执行异常: {e}", flush=True)
        return None


def model_ready(model):
    r = run([OLLAMA, "list"], timeout=20)
    if not r or r.returncode != 0:
        return False
    name = model.split(":")[0]
    for line in r.stdout.splitlines()[1:]:
        if line.strip().startswith(name):
            return True
    return False


def blob_bytes():
    total = 0
    try:
        for n in os.listdir(BLOBS):
            if "-partial" in n:
                p = os.path.join(BLOBS, n)
                try:
                    total += os.path.getsize(p)
                except OSError:
                    pass
    except FileNotFoundError:
        pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl:8b")
    ap.add_argument("--shutdown-delay", type=int, default=120)
    ap.add_argument("--no-shutdown", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(OLLAMA):
        print(f"[fatal] 找不到 ollama: {OLLAMA}")
        sys.exit(1)

    print(f"[start] 等待模型 {a.model}；ollama={OLLAMA}", flush=True)
    last = blob_bytes()
    last_change = time.time()
    started = time.time()

    while True:
        if model_ready(a.model):
            print(f"[done] 模型 {a.model} 已就绪，用时 {int(time.time()-started)}s", flush=True)
            break

        cur = blob_bytes()
        if cur != last:
            last, last_change = cur, time.time()
            print(f"[progress] {cur/1024/1024:.0f} MB  "
                  f"({time.strftime('%H:%M:%S')})", flush=True)

        if time.time() - last_change > STALL_SEC:
            print("[resume] 下载停滞，重跑 ollama pull（断点续传）", flush=True)
            run([OLLAMA, "pull", a.model], timeout=60 * 30)
            last_change = time.time()

        time.sleep(POLL)

    if a.no_shutdown:
        print("[skip] --no-shutdown，不关机", flush=True)
        return

    print(f"[power] {a.shutdown_delay}s 后关机；取消请运行: shutdown /a", flush=True)
    time.sleep(a.shutdown_delay)
    subprocess.run(["shutdown", "/s", "/t", "30"], check=False)
    print("[power] 已发出关机指令（30s 后执行）", flush=True)


if __name__ == "__main__":
    main()
