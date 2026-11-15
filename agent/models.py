#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型客户端：统一 chat(多模态) 接口。
- 默认 Ollama（兼容你现有的本地部署 qwen3vl:8b / ui-tars）
- 也支持任意 OpenAI 兼容后端（OpenAI / DeepSeek / 本地 vLLM 等）

两种后端的 messages 结构我们统一为 OpenAI 风格：
  messages = [
    {"role": "system", "content": "<str>"},
    {"role": "user",   "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}},
    ]},
  ]
注意：Ollama 原生 /api/chat 的图片格式不同——content 必须是纯文本，
图片要放在消息的 "images" 字段里（纯 base64，不带 data: 前缀）。
发送前由 _to_ollama_messages() 自动转换。
"""

import os
import json
import base64
import io
import urllib.request

from PIL import Image


def image_to_data_url(img: Image.Image) -> str:
    """PIL 图片 -> base64 data URL（PNG）。"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64


class ModelClient:
    def __init__(self, cfg: dict):
        p = cfg.get("provider", {}) or {}
        self.type = (os.environ.get("CU_PROVIDER") or p.get("type", "ollama")).lower()

        if self.type == "ollama":
            o = p.get("ollama", {}) or {}
            self.base_url = o.get("base_url", "http://localhost:11434/api/chat")
            self.model = os.environ.get("CU_MODEL") or o.get("model", "qwen3vl:8b")
            # 代码专用模型：纯文本任务（写代码）用 coder 版，同尺寸下代码能力明显更强；
            # 视觉任务仍用上面的 vl 版。8GB 显存装不下两个，Ollama 会按需换入换出（约 10s）。
            # 留空 = 不启用，所有任务都用主模型。
            self.coder_model = os.environ.get("CU_CODER_MODEL") or o.get("coder_model", "")
            self.api_key = None
        else:  # openai 兼容
            o = p.get("openai", {}) or {}
            self.base_url = (
                os.environ.get("CU_OPENAI_BASE_URL")
                or o.get("base_url", "https://api.openai.com/v1/chat/completions")
            )
            self.model = os.environ.get("CU_MODEL") or o.get("model", "gpt-4o")
            self.api_key = os.environ.get("CU_OPENAI_KEY") or o.get("api_key", "")

        self.temperature = float(os.environ.get("CU_TEMP") or cfg.get("agent", {}).get("temperature", 0.2))
        # 输出长度上限：本地小模型不设上限会一直往下瞎写（每次可吐满 4096 上下文，一步能跑一分钟）
        self.max_tokens = int(os.environ.get("CU_MAX_TOKENS") or cfg.get("agent", {}).get("max_tokens", 512))
        # 上下文窗口：Ollama 默认只有 4096，截图 + 历史一挤，
        # 「带名字的元素清单」根本塞不进去。提到 8192 才装得下。
        # 代价是 KV cache 多占约 0.5GB 显存（8GB 卡可承受）。
        self.num_ctx = int(os.environ.get("CU_NUM_CTX") or cfg.get("agent", {}).get("num_ctx", 8192))
        # 两个模型共用一张 8GB 卡，装不下第二份（视觉 5.8G + coder 4.7G）。
        # coder 只是写码/审查时的辅助模型，用完就别占着显存 30 分钟把主模型挤走；
        # 主视觉模型要常驻，否则通用循环第一步就得重新加载 5.8GB（10~20 秒）。
        ag = cfg.get("agent", {}) or {}
        self.keep_alive = ag.get("keep_alive", "30m")
        self.coder_keep_alive = ag.get("coder_keep_alive", "5m")

    def chat(self, messages: list, temperature: float = None, max_tokens: int = None,
             model: str = None, keep_alive: str = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        if self.type == "ollama":
            return self._chat_ollama(messages, temp, max_tokens or self.max_tokens,
                                     model, keep_alive)
        return self._chat_openai(messages, temp)

    def _model_exists(self, name: str) -> bool:
        """本地是否装了某个模型（查一次 /api/tags 并缓存，本地请求，1ms 级）。"""
        if not hasattr(self, "_tags_cache"):
            self._tags_cache = set()
            try:
                base = self.base_url.replace("/api/chat", "/api/tags")
                with urllib.request.urlopen(base, timeout=5) as r:
                    data = json.loads(r.read().decode("utf-8"))
                self._tags_cache = {m.get("name", "") for m in data.get("models", [])}
            except Exception:
                self._tags_cache = set()
        return name in self._tags_cache

    def chat_coder(self, messages: list, temperature: float = 0.1, max_tokens: int = None) -> str:
        """纯文本的写代码任务专用：优先 coder 模型，没装/不可用就静默退回主模型。

        为什么单独一条路：视觉模型被"看图"训练分走了一部分注意力，
        同尺寸的 coder 版在代码上更强；而写代码根本不需要看图。
        "没装就静默跳过"是故意的 —— 配置里留着 coder_model，用户哪天装了
        自动生效，不用改代码，也不会每步都报一次 404。
        """
        cm = getattr(self, "coder_model", "")
        if not cm or not self._model_exists(cm):
            return self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        try:
            return self.chat(messages, temperature=temperature, max_tokens=max_tokens,
                             model=cm, keep_alive=self.coder_keep_alive)
        except Exception as e:
            print(f"[模型] coder 模型「{cm}」调用失败（{e}），退回主模型 {self.model}")
            return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def _chat_ollama(self, messages, temp, max_tokens=None, model=None, keep_alive=None):
        payload = {
            "model": model or self.model,
            "messages": self._to_ollama_messages(messages),
            "stream": False,
            # 别让模型在两次任务之间被卸载。Ollama 默认 5 分钟空闲就扔掉 6.9GB
            # 权重，下一次调用要重新加载 10~20 秒 —— 演示时中间答两个问题再出题，
            # 第一句就会卡住。30 分钟足够覆盖任何一场演示。
            "keep_alive": keep_alive or self.keep_alive,
            "options": {"temperature": temp, "num_predict": max_tokens or self.max_tokens,
                        "num_ctx": self.num_ctx},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        # 记下这次请求的规模。prompt 一旦逼近 num_ctx，prefill 会突然变慢：
        # 真机 8033/8192 那次整条请求 29 秒，而平时只要 9 秒 —— 没这两个数
        # 就只能靠猜，而它恰恰是"同一任务里时而 10 秒时而 30 秒"的唯一解释。
        try:
            self.last_stats = {
                "prompt": int(result.get("prompt_eval_count") or 0),
                "gen": int(result.get("eval_count") or 0),
            }
        except Exception:
            self.last_stats = {}
        return result.get("message", {}).get("content", "")

    @staticmethod
    def _to_ollama_messages(messages):
        """OpenAI 风格消息 -> Ollama 原生格式：
        content 数组拍平成纯文本；image_url 的 data URL 拆出纯 base64 放进 images 字段。"""
        out = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                out.append({"role": m.get("role", "user"), "content": c})
                continue
            text_parts, images = [], []
            for part in c or []:
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        url = url.split(",", 1)[-1]  # 只留纯 base64
                    if url:
                        images.append(url)
            msg = {"role": m.get("role", "user"), "content": "\n".join(text_parts)}
            if images:
                msg["images"] = images
            out.append(msg)
        return out

    def _chat_openai(self, messages, temp):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.base_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        # 兼容 /v1/chat/completions 返回结构
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return result.get("message", {}).get("content", "")
