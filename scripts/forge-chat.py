#!/root/miniconda3/bin/python
"""Small dependency-free terminal client for the local forge-server."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "http://127.0.0.1:6006/v1/chat/completions"
KEY_FILE = Path("/root/.config/ggml-forge/api-key")
CONFIG_FILE = Path("/root/.config/ggml-forge/server.env")
DEFAULT_SYSTEM = (
    "你是一个友好、可靠的 Minecraft AI 玩家。使用中文自然地回答，"
    "不知道时就明确说明，不要编造游戏状态。"
)
MAX_HISTORY_CHARS = 300_000


def load_config() -> dict[str, str]:
    values: dict[str, str] = {}
    if not CONFIG_FILE.is_file():
        return values
    for raw_line in CONFIG_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def visible_text(content: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return cleaned.strip() or content.strip()


def trim_history(messages: list[dict[str, str]]) -> None:
    while len(messages) > 3 and sum(len(item.get("content", "")) for item in messages) > MAX_HISTORY_CHARS:
        del messages[1:3]


def complete(
    messages: list[dict[str, str]],
    api_key: str,
    api_url: str,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = json.dumps(
        {
            "model": "llm",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"cannot reach forge-server: {error.reason}") from error
    return result["choices"][0]["message"].get("content") or ""


def main() -> int:
    if not KEY_FILE.is_file():
        print(f"API key not found: {KEY_FILE}", file=sys.stderr)
        return 1
    api_key = KEY_FILE.read_text(encoding="utf-8").strip()
    config = load_config()
    host = config.get("FORGE_HOST", "127.0.0.1")
    port = int(config.get("FORGE_PORT", "6006"))
    api_url = f"http://{host}:{port}/v1/chat/completions"
    max_tokens = int(config.get("FORGE_CHAT_MAX_TOKENS", "2048"))
    temperature = float(config.get("FORGE_CHAT_TEMPERATURE", "0.7"))
    system_prompt = DEFAULT_SYSTEM
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print("GGML-Forge terminal chat")
    print("Commands: /reset, /system <prompt>, /exit")
    while True:
        try:
            user_text = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            print("再见。")
            return 0
        if user_text == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("会话历史已清空。")
            continue
        if user_text.startswith("/system "):
            system_prompt = user_text[len("/system "):].strip()
            messages = [{"role": "system", "content": system_prompt}]
            print("System prompt 已更新，会话历史已清空。")
            continue

        messages.append({"role": "user", "content": user_text})
        trim_history(messages)
        try:
            raw = complete(messages, api_key, api_url, max_tokens, temperature)
        except (RuntimeError, KeyError, ValueError) as error:
            messages.pop()
            print(f"请求失败：{error}", file=sys.stderr)
            continue
        messages.append({"role": "assistant", "content": raw})
        print(f"\nAI > {visible_text(raw)}")


if __name__ == "__main__":
    raise SystemExit(main())
