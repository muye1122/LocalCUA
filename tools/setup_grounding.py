#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载默认 grounding(yolo_caption) 需要的 icon_detect ONNX 权重。

models/ 被 .gitignore 排除，所以 clone 之后必须跑一次本脚本。
只用 Python 标准库：支持代理环境变量、断点续传、进度显示与基本完整性校验。
不会点击鼠标、不会打字，也不需要管理员权限。

用法：
    python tools/setup_grounding.py            # 已存在且 >1MB 就跳过
    python tools/setup_grounding.py --check   # 只看在不在，不联网
    python tools/setup_grounding.py --force   # 强制重新下载
    python tools/setup_grounding.py --no-verify
        # 仅在「网络有 TLS 中间代理、报证书错误」时使用，默认不需要
"""
import argparse
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models" / "icon_detect" / "model.onnx"
TMP = DEST.with_suffix(".onnx.part")
URL = ("https://www.modelscope.cn/models/onnx-community/"
       "OmniParser-v2.0_icon_detect/resolve/master/onnx/model.onnx")
MIN_SIZE = 1024 * 1024          # 小于 1MB 一定是错误页，不是权重


def ready():
    return DEST.exists() and DEST.stat().st_size > MIN_SIZE


def human(n):
    return "%.1f MB" % (n / 1024 / 1024)


def main():
    ap = argparse.ArgumentParser(description="下载 icon_detect ONNX 权重")
    ap.add_argument("--check", action="store_true", help="只检查，不联网")
    ap.add_argument("--force", action="store_true", help="强制重新下载")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过 TLS 证书校验（仅用于有中间代理的网络）")
    args = ap.parse_args()

    if args.check:
        if ready():
            print("OK  权重已就位：%s（%s）" % (DEST, human(DEST.stat().st_size)))
            return 0
        print("MISS 缺少权重：%s" % DEST)
        print("     跑一次 python tools/setup_grounding.py 即可")
        return 1

    if ready() and not args.force:
        print("已就位：%s（%s），跳过。想重下加 --force" % (DEST, human(DEST.stat().st_size)))
        return 0

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if TMP.exists() and TMP.stat().st_size < MIN_SIZE:
        TMP.unlink()                       # 上次可能只下到错误页

    start = TMP.stat().st_size if TMP.exists() else 0
    headers = {"User-Agent": "LocalCUA-setup/1.0"}
    if start:
        headers["Range"] = "bytes=%d-" % start
        print("断点续传：从 %s 继续" % human(start))
    else:
        print("下载 %s" % URL)

    ctx = None
    if args.no_verify:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    t0 = time.time()
    got = start
    milestone = got // (5 * 1024 * 1024)
    try:
        req = urllib.request.Request(URL, headers=headers)
        with urllib.request.urlopen(req, timeout=40, context=ctx) as resp:
            append = (start > 0 and getattr(resp, "status", 200) == 206)
            mode = "ab" if append else "wb"
            if not append:
                got = 0
            with open(TMP, mode) as f:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    m = got // (5 * 1024 * 1024)
                    if m != milestone:
                        milestone = m
                        print("  ... %s" % human(got))
    except Exception as e:
        print("下载失败：%s: %s" % (type(e).__name__, e))
        print("可手动下载到 %s：" % DEST)
        print("  " + URL)
        print("网络有 TLS 中间代理时可加 --no-verify 重试；已下的部分会保留，下次续传。")
        return 1

    size = TMP.stat().st_size
    if size < MIN_SIZE:
        print("文件只有 %d 字节，疑似错误页，已丢弃" % size)
        TMP.unlink(missing_ok=True)
        return 1
    TMP.replace(DEST)
    print("完成：%s（%s，耗时 %.1fs）" % (DEST, human(size), time.time() - t0))
    print("下一步：python tools/selfcheck.py 确认环境齐了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
