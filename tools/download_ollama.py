#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并发下载 Ollama Windows 安装包（绕过单连接 ~30KB/s 限速），下完自动静默安装。
改进点（相对初版）：
  - 40 片并发；任一缺失/体积不对就单独补下，循环到全齐为止（不会再半途崩）
  - 合并前逐片校验大小，合并写临时文件，校验通过才 rename，避免留半截
  - 每片用 curl（实测能正确透传 Range 过 ollama.com 的 307 重定向）
用法：python tools/download_ollama.py
"""
import os
import time
import sys
import subprocess
import concurrent.futures

URL = "https://ollama.com/download/OllamaSetup.exe"
PIN_IP = "34.36.133.15"   # ollama.com 能通的 IP，绕开抽风解析
RESOLVE = ["--resolve", f"ollama.com:443:{PIN_IP}"]
OUT = os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")),
                    "Downloads", "OllamaSetup.exe")
TMP = OUT + ".tmp"
CHUNKS = 40
ROUNDS = 100000     # 近乎无限重试：链路抽风时持续等恢复，连不上就下一轮
ROUND_GAP = 30      # 每轮之间歇 30s，避免无谓死磕
CURL_TMO = 240
CURL = "curl"


# 已知大小（v0.40.14，昨晚实测确认；网络抽风取不到动态大小时用此兜底）
KNOWN_SIZE = 1675999728


def total_size():
    # 快速探一次动态大小：短超时、最多 2 次，避免在被限速链路上干等
    for _ in range(2):
        try:
            r = subprocess.run(
                [CURL, "-s", "-r", "0-0", "--ssl-no-revoke", *RESOLVE, "-D", "-", "-o", "/dev/null", URL],
                capture_output=True, text=True, timeout=15)
            for line in r.stdout.splitlines():
                if line.lower().startswith("content-range:"):
                    total = int(line.strip().split("/")[-1])
                    if total > 100000000:
                        return total
        except Exception:
            pass
        time.sleep(2)
    # 兜底：HEAD 跟随重定向找 content-length（再试 2 次）
    for _ in range(2):
        try:
            r = subprocess.run([CURL, "-sIL", "--ssl-no-revoke", URL],
                               capture_output=True, text=True, timeout=15)
            for line in reversed(r.stdout.splitlines()):
                if line.lower().startswith("content-length:"):
                    total = int(line.split(":")[1].strip())
                    if total > 100000000:
                        return total
        except Exception:
            pass
        time.sleep(2)
    print(f"[WARN] 取不到动态大小，使用已知常量 KNOWN_SIZE={KNOWN_SIZE}", flush=True)
    return KNOWN_SIZE


def fetch_chunk(rng):
    """断点续传：保留已下字节，从断点续；连续多次无进展才放弃本轮。"""
    start, end, path = rng
    exp = end - start + 1
    seg = path + ".seg"
    stall = 0
    while True:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= exp:
            return True
        if have > exp:  # 异常：超出本片长度，重置
            with open(path, "wb"):
                pass
            have = 0
        need = exp - have
        r_start = start + have
        try:
            subprocess.run(
                [CURL, "-fL", "--ssl-no-revoke", *RESOLVE, "--connect-timeout", "15", "--max-time", "180",
                 "-r", f"{r_start}-{end}", URL, "-o", seg],
                capture_output=True, timeout=200,
            )
        except Exception:
            pass
        progressed = False
        if os.path.exists(seg):
            got = os.path.getsize(seg)
            if 0 < got <= need:      # 连续数据，追加（真正的续传）
                with open(path, "ab") as f, open(seg, "rb") as g:
                    f.write(g.read())
                progressed = True
            # got==0 或 got>need(服务端忽略 Range) 一律丢弃，避免污染
            os.remove(seg)
        after = os.path.getsize(path) if os.path.exists(path) else 0
        if after >= exp:
            return True
        if progressed and after > have:
            stall = 0
        else:
            stall += 1
            if stall >= 4:        # 连续 4 次没进展 → 本轮放弃，交由下一轮
                return False
        time.sleep(1)


def main():
    size = total_size()
    print(f"文件总大小: {size} bytes，分 {CHUNKS} 片并发下载", flush=True)
    step = size // CHUNKS
    ranges = []
    for i in range(CHUNKS):
        s = i * step
        e = size - 1 if i == CHUNKS - 1 else s + step - 1
        ranges.append((s, e, OUT + f".p{i:02d}"))

    for rnd in range(ROUNDS):
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=CHUNKS) as ex:
            ok = list(ex.map(fetch_chunk, ranges))
        missing = [i for i, v in enumerate(ok) if not v]
        got = sum(os.path.getsize(ranges[i][2]) for i in range(CHUNKS)
                  if os.path.exists(ranges[i][2]))
        el = time.time() - t0
        print(f"轮 {rnd+1}: 成功 {CHUNKS-len(missing)}/{CHUNKS}  已下 {got/1024/1024:.0f}MB"
              f"  本轮 {el:.0f}s" + (f"  缺片: {missing}" if missing else "  ✅ 全齐"), flush=True)
        if not missing:
            break
        if missing:
            print(f"  缺片 {missing}，{ROUND_GAP}s 后重试（链路可能抽风，持续等恢复）...", flush=True)
            time.sleep(ROUND_GAP)

    # 合并（写临时文件）
    print("合并分片...", flush=True)
    total = 0
    with open(TMP, "wb") as out:
        for i in range(CHUNKS):
            p = ranges[i][2]
            sz = os.path.getsize(p)
            with open(p, "rb") as f:
                out.write(f.read())
            total += sz
            os.remove(p)
    if total != size:
        print(f"[ERROR] 大小不匹配 {total} != {size}", flush=True)
        sys.exit(1)
    os.replace(TMP, OUT)
    print(f"下载完成，大小 {total} bytes ({total/1024/1024:.0f} MB)", flush=True)

    print("静默安装 Ollama...", flush=True)
    os.system(f'"{OUT}" /S')
    time.sleep(25)

    # 定位 ollama 可执行文件
    ollama = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
    if not os.path.exists(ollama):
        # 退而求其次：靠 PATH
        ollama = "ollama"
    print("ollama 路径:", ollama, flush=True)

    # 启动服务（后台）
    try:
        subprocess.Popen([ollama, "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception as e:
        print("serve 启动失败(可忽略，pull 会自动拉起):", e, flush=True)
    time.sleep(8)

    # 拉取视觉模型：一直重试到 24:00 或成功（避免没弄好就提前关机）
    import datetime
    model = "qwen2.5vl:7b"
    now = datetime.datetime.now()
    deadline = (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0)  # 今晚 24:00
    deadline_ts = deadline.timestamp()
    pulled = False
    attempt = 0
    while time.time() < deadline_ts:
        attempt += 1
        remain = int(deadline_ts - time.time())
        print(f"拉取模型 {model}（第 {attempt} 次，距 24:00 还有 {remain}s）...", flush=True)
        try:
            r = subprocess.run([ollama, "pull", model], capture_output=True, text=True,
                               timeout=min(1800, remain))
        except subprocess.TimeoutExpired:
            print("  本次超时，继续重试", flush=True)
            continue
        if r.returncode == 0:
            pulled = True
            print("模型拉取成功 ✅", flush=True)
            break
        print(f"  本次失败(rc={r.returncode})，30s 后重试", flush=True)
        time.sleep(30)

    print("DONE_EXE")

    if pulled:
        print("✅ 全部就绪：Ollama 已安装 + qwen2.5vl:7b 模型已就绪，本地视觉识别可用。", flush=True)
    else:
        # 没拉下来：Ollama 引擎已装好，可改用云端 API 跑 LocalCUA
        print("⚠️ Ollama 引擎已装好，但 qwen2.5vl:7b 模型因网络限速未拉取成功。"
              "可用云端 API（填 .env 的 CU_OPENAI_KEY）跑 LocalCUA，或网络好转后手动: ollama pull qwen2.5vl:7b", flush=True)


if __name__ == "__main__":
    main()
