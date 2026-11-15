#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 ModelScope（国内源）拉 GGUF 权重，再导入 ollama。

为什么不用 `ollama pull`：
  本机实测 registry.ollama.ai 只有 ~330KB/s，4.68GB 要 4 小时；
  ModelScope 走国内 CDN，单连接 1.5MB/s、4 连接约 2MB/s，同样的文件 ~40 分钟。
  而且 GGUF 是同一个权重（Qwen 官方仓库导出的 q4_k_m，字节数与 ollama 的
  4.68GB 层完全一致），导入后就是同一个模型。

特性：多连接分块 + 断点续传（每块一个 .partN 文件）+ sha256 校验 + 自动导入。

用法：
  python tools/fetch_gguf.py                      # 默认 qwen2.5-coder:7b
  python tools/fetch_gguf.py --conns 8            # 改并发数（实测 4 最划算）
  python tools/fetch_gguf.py --create qwen2.5-coder:7b   # 下完直接 ollama create
  python tools/fetch_gguf.py --check              # 只校验已有文件
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_REPO = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
MS = "https://modelscope.cn"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def api_files(repo, revision="master"):
    url = f"{MS}/api/v1/models/{repo}/repo/files?Revision={revision}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))["Data"]["Files"]


def resolve_url(repo, filename, revision="master"):
    """拿 CDN 直链（带 auth_key，有效期几小时 —— 续传前要重新解析）。"""
    url = f"{MS}/models/{repo}/resolve/{revision}/{urllib.parse.quote(filename)}"
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        final = r.geturl()
        size = int(r.headers.get("Content-Length") or 0)
    return final, size


def download_range(url, start, end, dst, stop, label, retries=6):
    """把 [start, end] 下到 dst，支持续传：dst 已有 n 字节就从 start+n 接着下。"""
    for attempt in range(1, retries + 1):
        have = dst.stat().st_size if dst.exists() else 0
        if have >= end - start + 1:
            return True
        if stop.is_set():
            return False
        try:
            headers = dict(UA)
            headers["Range"] = f"bytes={start + have}-{end}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                if r.status not in (200, 206):
                    raise IOError(f"HTTP {r.status}")
                with open(dst, "ab") as f:
                    while True:
                        if stop.is_set():
                            return False
                        chunk = r.read(1 << 18)
                        if not chunk:
                            break
                        f.write(chunk)
            if dst.stat().st_size >= end - start + 1:
                return True
            raise IOError(f"段不完整 {dst.stat().st_size}/{end - start + 1}")
        except Exception as e:
            if stop.is_set():
                return False
            print(f"  [{label}] 第{attempt}次中断：{str(e)[:70]}，2s 后续传")
            stop.wait(2)
    return False


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def fetch(repo, filename, out_dir, conns, want_sha):
    files = {f["Name"]: f for f in api_files(repo)}
    if filename not in files:
        print(f"仓库里没有 {filename}，可选：")
        for n in sorted(files):
            if n.endswith(".gguf"):
                print("  ", n)
        return None
    meta = files[filename]
    size = int(meta["Size"])
    sha = meta.get("Sha256") or want_sha
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / filename

    print(f"目标：{repo}/{filename}")
    print(f"大小：{size/1e9:.2f} GB   源：ModelScope（国内 CDN）   并发：{conns}")

    if dst.exists() and dst.stat().st_size == size:
        print("文件已存在且大小一致，直接校验")
        return dst, sha

    # 分块，每块一个 .partN。块数固定，重跑时按块续传，不必从头来。
    # 块数一变，各块的字节边界就全变了，已有的 .part 会变成垃圾数据 ——
    # 所以把（块数, 大小）记在边上，续传时对不上就拦住，而不是默默拼出错文件。
    meta_path = out_dir / f"{filename}.chunks.json"
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        if old and (int(old.get("conns", conns)) != conns or int(old.get("size", size)) != size):
            print(f"⚠️ 这个文件上次是用 {old.get('conns')} 个分块下的，"
                  f"现在改成 {conns} 会作废已下的数据。")
            print(f"   请用同一条命令续传：--conns {old.get('conns')}")
            return None
    else:
        meta_path.write_text(json.dumps({"conns": conns, "size": size}), encoding="utf-8")

    per = size // conns
    segs = []
    for i in range(conns):
        s = i * per
        e = size - 1 if i == conns - 1 else (i + 1) * per - 1
        segs.append((s, e, out_dir / f"{filename}.part{i}"))

    url, _ = resolve_url(repo, filename)
    stop = threading.Event()
    done = [0]
    lock = threading.Lock()

    def worker(idx, s, e, path):
        ok = download_range(url, s, e, path, stop, f"段{idx}")
        if not ok:
            stop.set()

    ts = [threading.Thread(target=worker, args=(i, s, e, p), daemon=True)
          for i, (s, e, p) in enumerate(segs)]
    t0 = time.time()
    last = 0.0
    for t in ts:
        t.start()
    try:
        while any(t.is_alive() for t in ts):
            time.sleep(0.5)
            cur = sum(p.stat().st_size if p.exists() else 0 for _, _, p in segs)
            now = time.time()
            if now - last >= 2:
                last = now
                el = now - t0
                spd = cur / el if el > 0 else 0
                eta = (size - cur) / spd if spd > 0 else 0
                pct = cur * 100.0 / size
                bar = "#" * int(pct // 4)
                print(f"\r  {pct:5.1f}% [{bar:<25}] {human(cur)}/{human(size)} "
                      f"{spd/1e6:5.2f}MB/s 剩 {int(eta//60)}分{int(eta%60)}秒   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n中断：已完成的分块都留在磁盘上，重跑同一条命令接着下")
        stop.set()
        for t in ts:
            t.join(timeout=5)
        return None
    print()
    if stop.is_set():
        print("有分块没下完，重跑同一条命令会从断点继续")
        return None

    # 按顺序合并
    print("合并分块…")
    with open(dst, "wb") as f:
        for _, _, p in segs:
            with open(p, "rb") as g:
                while True:
                    b = g.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
    for _, _, p in segs:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        meta_path.unlink()
    except OSError:
        pass

    if dst.stat().st_size != size:
        print(f"❌ 合并后大小不对：{dst.stat().st_size} != {size}")
        return None
    print(f"✅ 下载完成：{dst}（{human(size)}，耗时 {int(time.time()-t0)//60} 分钟）")
    return dst, sha


def import_to_ollama(gguf_path, model_name, num_ctx=8192):
    """ollama create 需要 Modelfile。Qwen2.5 的对话模板是 ChatML，照这个写就与官方一致。"""
    mf = gguf_path.parent / f"Modelfile.{model_name.replace(':', '_').replace('/', '_')}"
    mf.write_text(
        f'FROM {gguf_path.as_posix()}\n'
        'TEMPLATE """{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
        '{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}'
        '<|im_start|>assistant\n"""\n'
        'PARAMETER stop <|im_start|>\n'
        'PARAMETER stop <|im_end|>\n'
        f'PARAMETER num_ctx {num_ctx}\n',
        encoding="utf-8")
    print(f"导入 ollama：{model_name}（Modelfile: {mf}）")
    p = subprocess.run(["ollama", "create", model_name, "-f", str(mf)],
                       capture_output=True, text=True)
    print(p.stdout.strip() or p.stderr.strip())
    if p.returncode != 0:
        print("❌ ollama create 失败")
        return False
    print(f"✅ 已导入。验证：ollama run {model_name} \"写一个求阶乘的C函数\"")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--out", default=str(Path(os.environ.get("USERPROFILE", ".")) / "models"))
    ap.add_argument("--conns", type=int, default=4)
    ap.add_argument("--create", default="", help="下完后 ollama create 成这个名字")
    ap.add_argument("--check", action="store_true", help="只校验已有文件")
    args = ap.parse_args()

    out = Path(args.out)
    dst = out / args.file
    if args.check:
        if not dst.exists():
            print(f"没有 {dst}")
            return 1
        files = {f["Name"]: f for f in api_files(args.repo)}
        sha = files[args.file]["Sha256"]
        print(f"计算 sha256（{human(dst.stat().st_size)}，约 20 秒）…")
        got = sha256_file(dst)
        ok = got == sha
        print(("✅ 校验通过 " if ok else "❌ 校验不符 ") + got[:16])
        return 0 if ok else 1

    r = fetch(args.repo, args.file, out, max(1, args.conns), None)
    if not r:
        return 1
    dst, sha = r
    if sha:
        print("校验 sha256（约 20 秒）…")
        got = sha256_file(dst)
        if got != sha:
            print(f"❌ sha256 不符\n  期望 {sha}\n  实际 {got}")
            print("   文件保留，可用 --check 复查；确认坏文件请手动删除后重下")
            return 1
        print("✅ sha256 一致")
    if args.create:
        import_to_ollama(dst, args.create)
    return 0


if __name__ == "__main__":
    sys.exit(main())
