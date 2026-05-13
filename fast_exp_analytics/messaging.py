from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

@dataclass
class ChatSendConfig:
    api_base_url: str
    token: str
    timeout: int = 30
    parse_mode: str = "HTML"

def send_chat_message(*, config: ChatSendConfig, message: str, chat_id: str, file_path: str | None = None, file_name: str | None = None) -> dict[str, Any]:
    if not config.token:
        raise ValueError("config.token is required")
    base_url = config.api_base_url.rstrip("/")
    if not file_path:
        url = f"{base_url}/messages/sendText"
        payload = {"token": config.token, "chatId": chat_id, "text": message, "parseMode": config.parse_mode}
        resp = requests.post(url, data=payload, timeout=config.timeout)
        try:
            data = resp.json()
        except Exception:
            data = {"status_code": resp.status_code, "text": resp.text}
        if resp.status_code != 200:
            raise RuntimeError(f"Text send failed: {resp.status_code} {resp.text}")
        return data

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    url = f"{base_url}/messages/sendFile"
    payload = {"token": config.token, "chatId": chat_id, "caption": message, "parseMode": config.parse_mode}
    send_name = file_name or os.path.basename(file_path)
    with open(file_path, "rb") as f:
        files = {"file": (send_name, f)}
        resp = requests.post(url, data=payload, files=files, timeout=config.timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"status_code": resp.status_code, "text": resp.text}
    if resp.status_code != 200:
        raise RuntimeError(f"File send failed: {resp.status_code} {resp.text}")
    return data
