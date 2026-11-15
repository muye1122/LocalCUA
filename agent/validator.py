#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验器：每次动作后回截屏，让 VLM 判断“点中没/成功了没”，失败可触发重规划。

注意：开启会多一次模型调用，更慢但更稳。默认关闭，按需开启（config.yaml）。
"""

from agent.models import ModelClient, image_to_data_url


class Validator:
    def __init__(self, cfg: dict, model: ModelClient):
        v = (cfg.get("validator", {}) or {})
        self.enabled = bool(v.get("enabled", False))
        self.model = model

    def verify(self, intended_text: str, before_img, after_img):
        """返回 (success: bool, note: str)。未开启时直接返回成功。"""
        if not self.enabled:
            return True, ""
        prompt = (
            f"Agent 意图执行的动作是：{intended_text}\n"
            "第一张是动作前截图，第二张是动作后截图。\n"
            "请判断该动作是否成功达成预期。只回答 YES 或 NO，并简短说明原因。"
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(before_img)}},
            {"type": "image_url", "image_url": {"url": image_to_data_url(after_img)}},
        ]
        try:
            r = self.model.chat([{"role": "user", "content": content}], temperature=0.0)
        except Exception as e:
            return True, f"校验调用失败: {e}"
        ok = r.strip().upper().startswith("YES")
        return ok, r.strip()
