#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_model —— 从 Ollama registry 可靠下载模型（支持断点续传 + 完整性校验）。

为什么不用 `ollama pull`：
  实测 ollama 0.40.14 在这台机器上下 qwen2.5-coder:7b 会卡在分块收尾，
  最后留下一个**大小正确但内容全空**的稀疏文件（sha256 对不上），
  再执行 pull 又会从头重来。所以这里自己下：单连接、可续传、校验到底。

用法（随时中断，重跑会接着下）：
  python tools/fetch_model.py qwen2.5-coder:7b
  python tools/fetch_model.py qwen2.5-coder:7b --host registry.ollama.ai
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA_MODELS = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))


def http_get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_manifest(host, name_tag):
    name, _, tag = name_tag.partition(":")
    tag = tag or "latest"
    url = f"https://{host}/v2/library/{name}/manifests/{tag}"
    with http_get(url, {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}) as r:
        return json.loads(r.read().decode("utf-8"))


def blob_path(digest):
    return OLLAMA_MODELS / "blobs" / digest.replace(":", "-")


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_blob(host, digest, size, label=""):
    """单连接下载，支持断点续传。返回 True 表示最终校验通过。"""
    dst = blob_path(digest)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == size:
        print(f"  [{label}] 已存在且大小正确，校验中…", flush=True)
        if sha256_file(dst) == digest.split(":")[1]:
            print(f"  [{label}] ✅ 校验通过")
            return True
        print(f"  [{label}] ⚠️ 哈希不符，重新下载")
        dst.unlink()

    url = f"https://{host}/v2/library/{_name}/blobs/{digest}"
    have = dst.stat().st_size if dst.exists() else 0
    if have and have > size:
        have = 0
        dst.unlink()
    if have:
        print(f"  [{label}] 续传：已有 {have / 1e6:.1f}MB / {size / 1e6:.1f}MB")
    t0, last = time.time(), 0.0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        with http_get(url, headers, timeout=60) as r, open(dst, "ab" if have else "wb") as f:
            done = have
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 3:
                    last = now
                    sp = done / max(1e-6, now - t0)
                    eta = (size - done) / max(1.0, sp)
                    pct = done * 100.0 / size
                    print(f"  [{label}] {pct:5.1f}%  {done / 1e6:7.1f}/{size / 1e6:.0f}MB  "
                          f"{sp / 1e3:6.0f}KB/s  剩余 {eta / 60:5.1f} 分钟", flush=True)
    except KeyboardInterrupt:
        print(f"\n  [{label}] 已中断，文件保留，重跑本命令会接着下。")
        raise
    except Exception as exc:
        print(f"\n  [{label}] 下载中断（{exc}）。已下部分保留，重跑会续传。")
        return False

    got = dst.stat().st_size
    if got != size:
        print(f"  [{label}] 大小不符 {got} != {size}")
        return False
    print(f"  [{label}] 校验中…", flush=True)
    if sha256_file(dst) != digest.split(":")[1]:
        print(f"  [{label}] ❌ 哈希不符，文件已删除，请重跑")
        dst.unlink(missing_ok=True)
        return False
    print(f"  [{label}] ✅ 完成")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="例如 qwen2.5-coder:7b")
    ap.add_argument("--host", default="registry.ollama.ai")
    args = ap.parse_args()

    global _name
    name, _, tag = args.model.partition(":")
    tag = tag or "latest"
    _name = name

    print(f"[1/3] 取 manifest：{name}:{tag}")
    man = fetch_manifest(args.host, args.model)
    layers = man.get("layers", [])
    cfg = man.get("config", {})
    total = sum(int(x.get("size", 0)) for x in layers) + int(cfg.get("size", 0))
    print(f"      共 {len(layers)} 层 + config，总 {(total) / 1e9:.2f}GB")

    print("[2/3] 下载各层")
    todo = [cfg] + layers
    for i, item in enumerate(todo):
        d, s = item["digest"], int(item.get("size", 0))
        mt = item.get("mediaType", "").rsplit(".", 1)[-1]
        ok = download_blob(args.host, d, s, f"{i + 1}/{len(todo)} {mt}")
        if not ok:
            print("\n❌ 未完成。网络恢复后重跑同样命令即可续传。")
            return 1

    print("[3/3] 写 manifest")
    mdir = OLLAMA_MODELS / "manifests" / "registry.ollama.ai" / "library" / name
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / tag).write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    print(f"      {mdir / tag}")
    print(f"\n✅ 完成。现在可以直接用：ollama run {name}:{tag}   （或 ollama list 确认）")
    return 0


_name = ""

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
