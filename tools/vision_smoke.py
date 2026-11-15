"""视觉冒烟测试：截屏 → 发给本地 Ollama 视觉模型 → 看它能否描述并给出元素坐标。

用法: python tools/vision_smoke.py
"""

import base64
import io
import json
import time
import urllib.request

import pyautogui
from PIL import Image

API = "http://localhost:11434/api/chat"
MODEL = "qwen2.5vl:7b"
MAXW = 1280


def screenshot_b64():
    img = pyautogui.screenshot()
    if img.width > MAXW:
        h = int(img.height * MAXW / img.width)
        img = img.resize((MAXW, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), img.size


def ask(img_b64, prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 400},
    }
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode())
    return d["message"]["content"], time.time() - t0


def main():
    b64, size = screenshot_b64()
    print(f"[shot] 截图 {size[0]}x{size[1]}，base64 {len(b64)//1024} KB", flush=True)
    prompt = (
        "这是一张 Windows 桌面截图。请回答两点：\n"
        "1) 屏幕上现在有什么（一句话）；\n"
        "2) 列出 3 个最明显的可点击元素，每个用 JSON 格式给出 {名称, x, y}，"
        "坐标用图片像素（左上角为 0,0）。"
    )
    text, dt = ask(b64, prompt)
    print(f"[reply] 用时 {dt:.1f}s\n{text}", flush=True)


if __name__ == "__main__":
    main()
