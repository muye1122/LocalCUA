#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web UI：FastAPI 服务 + 自包含 HTML 页面 + WebSocket。

功能：
  - 实时日志流
  - 下达任务 / 停止
  - 高风险操作弹出「允许/拒绝」确认
  - 模型 ask 时弹出回复框
全部本地运行，无外部 CDN 依赖。
"""

import asyncio
import json
import os
import queue
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from agent.config import load_config
from agent.core import AgentSession

CFG = load_config()


class AgentController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stop_event = threading.Event()
        self.log_q: "queue.Queue" = queue.Queue()
        self.confirm_event = threading.Event()
        self.confirm_result = False
        self.busy = False
        self.awaiting_reply = False
        self.current_task = ""
        self.worker = None

    def _log(self, msg):
        self.log_q.put({"type": "log", "msg": msg})

    def _confirm(self, action):
        self.log_q.put({"type": "confirm_request", "action": action})
        self.confirm_event.clear()
        self.confirm_event.wait()  # 阻塞，等 Web 客户端回复
        return self.confirm_result

    def _new_session(self):
        return AgentSession(self.cfg, log=self._log, confirm=self._confirm,
                            stop_event=self.stop_event)

    def start_task(self, text):
        if self.busy:
            # 真机现象：agent 在控制台页面上误点「开始」，任务被重复提交 6 次，
            # 日志里一串"任务进行中"其实是它自己点出来的。说清来源免得误导排查。
            self._log("⚠️ 已有任务在跑，本次提交已忽略（常见原因：agent 把鼠标点到了本页面）。")
            return
        if not text.strip():
            return
        self.current_task = text
        self.stop_event.clear()
        self.awaiting_reply = False
        sess = self._new_session()
        self.session = sess
        self.busy = True          # 先占位再起线程，否则连点两下会起两个 worker
        self.worker = threading.Thread(
            target=self._run, args=(sess, text, None), daemon=True
        )
        self.worker.start()

    def resume_reply(self, text):
        if self.busy:
            self._log("⚠️ 正在处理，稍候。")
            return
        sess = getattr(self, "session", None) or self._new_session()
        self.worker = threading.Thread(
            target=self._run, args=(sess, self.current_task, text), daemon=True
        )
        self.worker.start()

    def _run(self, sess, task, reply):
        self.busy = True
        try:
            result = sess.run(task, reply=reply)
            if result == "NEED_REPLY":
                self.awaiting_reply = True
                self.log_q.put({"type": "need_reply"})
            else:
                self.log_q.put({"type": "finished", "result": result})
        except Exception as e:
            self.log_q.put({"type": "error", "msg": str(e)})
        finally:
            self.busy = False

    def handle_msg(self, msg):
        t = msg.get("type")
        if t == "task":
            self.start_task(msg.get("text", ""))
        elif t == "stop":
            self.stop_event.set()
            self._log("⛔ 已发送停止信号。")
        elif t == "confirm":
            self.confirm_result = bool(msg.get("allow", False))
            self.confirm_event.set()
        elif t == "reply":
            self.resume_reply(msg.get("text", ""))


controller = AgentController(CFG)

app = FastAPI(title="computer-use-agent v2")

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>本地电脑操作 Agent v2</title>
<style>
  :root{ --bg:#f7f8fa; --panel:#fff; --border:#e3e6ea; --text:#1f2329; --muted:#8a9099;
         --accent:#2563eb; --danger:#dc2626; --ok:#16a34a; }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,"Microsoft YaHei",Segoe UI,Roboto,sans-serif;
       background:var(--bg);color:var(--text)}
  .wrap{max-width:880px;margin:0 auto;padding:18px}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:14px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:14px}
  textarea{width:100%;min-height:64px;resize:vertical;padding:10px;border:1px solid var(--border);
           border-radius:8px;font-size:14px;font-family:inherit}
  .row{display:flex;gap:10px;margin-top:10px;align-items:center}
  button{cursor:pointer;border:none;border-radius:8px;padding:9px 16px;font-size:14px;color:#fff}
  .primary{background:var(--accent)} .danger{background:var(--danger)} .ghost{background:#eef1f5;color:var(--text)}
  #log{height:320px;overflow:auto;background:#0f1115;color:#d6e2ff;padding:10px;border-radius:8px;
       font-family:Consolas,Menlo,monospace;font-size:12.5px;white-space:pre-wrap;line-height:1.5}
  /* 模型自报的思路单独一栏：就是"看得见它在想什么"的那个位置 */
  #log .think{color:#9cc2ff;font-style:italic;border-left:2px solid #3b6fd4;padding:2px 0 2px 8px;
              margin:4px 0;background:rgba(59,111,212,.10);border-radius:0 4px 4px 0}
  #log .seq{color:#8ee6a8}
  .status{font-size:13px;color:var(--muted);margin-top:8px}
  .modal{position:fixed;inset:0;background:rgba(0,0,0,.35);display:none;align-items:center;justify-content:center}
  .modal .box{background:#fff;border-radius:12px;padding:20px;max-width:460px;width:90%}
  .modal pre{background:#f3f4f6;padding:10px;border-radius:8px;overflow:auto;font-size:12px}
  .hidden{display:none}
</style>
</head>
<body>
<div class="wrap">
  <h1>本地电脑操作 Agent · v2</h1>
  <div class="sub">模型：__MODEL__ ｜ grounding：__GROUNDING__ ｜ 规划：__PLANNER__ ｜ 校验：__VALIDATOR__</div>

  <div class="card">
    <textarea id="task" placeholder="例如：打开计算器，算 123+456，把结果记下来"></textarea>
    <div class="row">
      <button class="primary" id="start">▶ 开始</button>
      <button class="danger" id="stop">■ 停止</button>
      <span class="status" id="status">就绪</span>
    </div>
  </div>

  <div class="card">
    <div id="log"></div>
  </div>

  <div class="card hidden" id="replyCard">
    <div style="font-size:14px;margin-bottom:6px">模型在等你回复：</div>
    <textarea id="reply" placeholder="在这里写下你的回答…"></textarea>
    <div class="row"><button class="primary" id="sendReply">↩ 回复</button></div>
  </div>
</div>

<div class="modal" id="confirmModal">
  <div class="box">
    <div style="font-size:15px;font-weight:600;margin-bottom:8px">⚠️ 高风险操作确认</div>
    <div style="font-size:13px;color:#555;margin-bottom:8px">Agent 准备执行以下操作，是否允许？</div>
    <pre id="confirmJson"></pre>
    <div class="row">
      <button class="primary" id="allow">允许</button>
      <button class="danger" id="deny">拒绝</button>
    </div>
  </div>
</div>

<script>
const logEl=document.getElementById('log');
const statusEl=document.getElementById('status');
const replyCard=document.getElementById('replyCard');
const confirmModal=document.getElementById('confirmModal');
function addLog(m){const d=document.createElement('div');d.textContent=m;
  if(m.startsWith('[思考]'))d.className='think';
  else if(m.startsWith('[序列]'))d.className='seq';
  logEl.appendChild(d);logEl.scrollTop=logEl.scrollHeight;}
const ws=new WebSocket('ws://'+location.host+'/ws');
ws.onopen=()=>addLog('🔌 已连接');
ws.onmessage=(e)=>{const m=JSON.parse(e.data);
  if(m.type==='log'){addLog(m.msg);}
  else if(m.type==='confirm_request'){document.getElementById('confirmJson').textContent=JSON.stringify(m.action,null,2);confirmModal.style.display='flex';}
  else if(m.type==='need_reply'){replyCard.classList.remove('hidden');statusEl.textContent='等待你的回复';}
  else if(m.type==='finished'){statusEl.textContent='完成：'+m.result;addLog('—— 结束（'+m.result+'）');}
  else if(m.type==='error'){statusEl.textContent='出错';addLog('❌ '+m.msg);}
};
ws.onclose=()=>addLog('🔌 连接关闭');
function waitWs(){return ws.readyState===1;}
document.getElementById('start').onclick=()=>{const t=document.getElementById('task').value.trim();
  if(!t){return;}statusEl.textContent='运行中…';ws.send(JSON.stringify({type:'task',text:t}));};
document.getElementById('stop').onclick=()=>{ws.send(JSON.stringify({type:'stop'}));};
document.getElementById('sendReply').onclick=()=>{const r=document.getElementById('reply').value.trim();
  replyCard.classList.add('hidden');ws.send(JSON.stringify({type:'reply',text:r}));statusEl.textContent='运行中…';};
document.getElementById('allow').onclick=()=>{confirmModal.style.display='none';ws.send(JSON.stringify({type:'confirm',allow:true}));};
document.getElementById('deny').onclick=()=>{confirmModal.style.display='none';ws.send(JSON.stringify({type:'confirm',allow:false}));};
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    model = CFG.get("provider", {}).get("ollama", {}).get("model", "qwen3vl:8b")
    grounding = CFG.get("grounding", {}).get("backend", "none")
    planner = "开" if CFG.get("planner", {}).get("enabled", True) else "关"
    validator = "开" if CFG.get("validator", {}).get("enabled", False) else "关"
    html = (INDEX_HTML
            .replace("__MODEL__", str(model))
            .replace("__GROUNDING__", str(grounding))
            .replace("__PLANNER__", planner)
            .replace("__VALIDATOR__", validator))
    return HTMLResponse(html)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    sender_task = None

    async def sender():
        while True:
            try:
                item = controller.log_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            try:
                await websocket.send_json(item)
            except Exception:
                break

    sender_task = asyncio.create_task(sender())
    try:
        while True:
            msg = await websocket.receive_json()
            controller.handle_msg(msg)
    except WebSocketDisconnect:
        pass
    finally:
        if sender_task:
            sender_task.cancel()


def main(host=None, port=None):
    web = CFG.get("web", {}) or {}
    host = host or web.get("host", "127.0.0.1")
    port = int(port or web.get("port", 7860))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
