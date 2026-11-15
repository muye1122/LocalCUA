"""一键自检：只读不点，检查运行环境并把结论写成一份人可读的报告。

用途：你回到电脑前，双击 selfcheck.bat，它会把「哪一环断了」直接指出来，
不用再靠猜。整个过程不会移动鼠标、不会输入任何字符。

用法：
    python -m tools.selfcheck            # 中文报告到屏幕 + 落盘
产物：
    data/selfcheck/<时间戳>/report.txt   完整结论
    data/selfcheck/<时间戳>/screen.png   当前屏幕截图
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []  # (等级, 标题, 详情)  等级：OK / WARN / FAIL


def check(level, title, detail=""):
    RESULTS.append((level, title, detail))
    icon = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[level]
    print(f"{icon} {title}")
    if detail:
        for line in str(detail).splitlines():
            print(f"       {line}")


def main():
    out_dir = os.path.join("data", "selfcheck", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 68)
    print("LocalCUA 自检（不会点击鼠标、不会输入文字）")
    print(f"报告目录：{out_dir}")
    print("=" * 68)

    # 1) Python 与依赖
    try:
        import PIL
        import numpy

        check("OK", f"Python {sys.version.split()[0]} 可用，Pillow/NumPy 已装")
    except Exception as e:
        check("FAIL", "Python 依赖缺失", e)
        return finish(out_dir)

    # 2) 项目本身能否导入（配置、核心类）
    try:
        from agent.config import load_config
        from agent.core import AgentSession  # noqa: F401
        from agent.safety import ensure_dpi_aware

        ensure_dpi_aware()
        cfg = load_config()
        backend = (cfg.get("grounding") or {}).get("backend", "?")
        check("OK", f"项目模块导入正常，grounding.backend = {backend}")
    except Exception as e:
        check("FAIL", "项目模块导入失败", e)
        return finish(out_dir)

    # 3) 截屏能力（这是「能不能看到桌面」的前提）
    screen_path = os.path.join(out_dir, "screen.png")
    from PIL import ImageGrab

    try:
        t0 = time.time()
        img = ImageGrab.grab()
        cost = (time.time() - t0) * 1000
        img.save(screen_path)
        w, h = img.size
        extrema = img.convert("L").getextrema()
        if extrema[0] == extrema[1]:
            check("FAIL", f"截图是全黑的（{w}x{h}）",
                  "说明当前进程不在你的桌面会话里，或被锁屏挡住")
        else:
            check("OK", f"截屏正常 {w}x{h}，耗时 {cost:.0f}ms",
                  f"已保存 {screen_path}")
    except Exception as e:
        check("FAIL", "截屏失败", e)
        img = None

    # 4) OLLAMA 本地服务与模型
    model_name = (((cfg.get("provider") or {}).get("ollama") or {})
                  .get("model")) or "qwen3vl:8b"
    try:
        import requests

        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        tags = r.json().get("models") or []
        names = [m.get("name", "") for m in tags]
        if names:
            hit = [n for n in names if n.split(":")[0] == model_name.split(":")[0]]
            if hit:
                check("OK", f"Ollama 在跑，找到模型 {hit[0]}", f"共 {len(names)} 个模型")
            else:
                check("WARN", f"Ollama 在跑，但没找到 {model_name}",
                      "已有：" + ", ".join(names[:6]))
        else:
            check("WARN", "Ollama 在跑，但一个模型都没有", "先 ollama pull " + model_name)
    except Exception as e:
        check("FAIL", "连不上 Ollama (127.0.0.1:11434)",
              f"{e}\n→ 先启动 Ollama，再重跑自检")

    # 5) 视觉定位（ONNX），这一步决定「眼睛」是否好使
    onnx_path = os.path.join("models", "icon_detect", "model.onnx")
    if not os.path.exists(onnx_path):
        check("FAIL", "缺少元素检测权重",
              f"{onnx_path} 不存在，默认 yolo_caption 无法初始化。\n"
              "→ 跑一次 python tools/setup_grounding.py 自动下载（约 80MB）\n"
              "→ 或把 config.yaml 的 grounding.backend 改成 vlm 跳过这一步")
    elif img is not None:
        try:
            from agent.yolo_grounding import detect_elements

            t0 = time.time()
            dets = detect_elements(img, conf=0.3)
            cost = (time.time() - t0) * 1000
            if dets:
                check("OK", f"视觉检测可用：{len(dets)} 个元素，{cost:.0f}ms/帧",
                      "前 5 个：" + ", ".join(
                          f"({d[0]:.0f},{d[1]:.0f})-({d[2]:.0f},{d[3]:.0f})" for d in dets[:5]))
            else:
                check("WARN", "视觉模型能跑，但这张截图没检出元素", "可能整屏纯色或截图异常")
        except Exception as e:
            check("FAIL", "视觉检测运行失败", e)

    # 6) 前台窗口（判断焦点是否正常，也确认微信这类目标程序在不在）
    try:
        import ctypes

        u = ctypes.windll.user32
        hwnd = u.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(512)
        u.GetWindowTextW(hwnd, buf, 512)
        title = buf.value or "(无标题)"
        check("OK", "前台窗口可读", title)
    except Exception as e:
        check("WARN", "读不到前台窗口", e)

    try:
        import subprocess

        out = subprocess.run(["tasklist"], capture_output=True, text=True, errors="ignore").stdout.lower()
        for proc, label in (("weixin.exe", "微信"), ("wechat.exe", "微信"),
                            ("notepad.exe", "记事本")):
            if proc in out:
                check("OK", f"{label} 正在运行（{proc}）")
    except Exception:
        pass

    return finish(out_dir)


def finish(out_dir):
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print("=" * 68)
    if fails:
        print(f"结论：{len(fails)} 项失败、{len(warns)} 项提醒 —— 见上面 [FAIL] 的条目")
    elif warns:
        print(f"结论：环境基本可用，{len(warns)} 项提醒")
    else:
        print("结论：全部通过，可以直接跑任务")
    print("=" * 68)

    report = os.path.join(out_dir, "report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"LocalCUA 自检报告  {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for level, title, detail in RESULTS:
            f.write(f"[{level}] {title}\n")
            if detail:
                f.write("".join(f"    {ln}\n" for ln in str(detail).splitlines()))
        f.write("\n")
    print(f"报告已写入：{report}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
