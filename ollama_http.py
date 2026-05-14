import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


class OllamaError(RuntimeError):
    pass


def _request_json(method: str, url: str, payload: Optional[Dict[str, Any]], timeout_s: float):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise OllamaError(f"HTTP {e.code} {e.reason}: {body}") from e
    except urllib.error.URLError as e:
        raise OllamaError(
            f"无法连接 Ollama：{e.reason}（url={url}）。请确认 Ollama 已启动，并且 base_url 可访问。"
        ) from e


def list_models(base_url: str = DEFAULT_BASE_URL, timeout_s: float = 10.0) -> List[str]:
    url = f"{base_url}/api/tags"
    data = _request_json("GET", url, None, timeout_s=timeout_s) or {}
    models = data.get("models") or []
    names = []
    for m in models:
        n = m.get("name")
        if n:
            names.append(n)
    return names


def pick_gemma_model(preferred: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> str:
    models = list_models(base_url=base_url)
    if preferred:
        if preferred in models:
            return preferred
        raise OllamaError(f"未找到模型：{preferred}。当前本地模型：{models or '（空）'}")
    for name in models:
        if name.lower().startswith("gemma"):
            return name
    raise OllamaError(
        "未在本地发现 gemma 模型。请先执行例如：ollama pull gemma2:2b（或你想用的 gemma 变体），然后重试。"
    )


def chat(
    *,
    model: str,
    system: str,
    user: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 600.0,
) -> str:
    url = f"{base_url}/api/chat"
    options: Dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": options,
    }
    data = _request_json("POST", url, payload, timeout_s=timeout_s) or {}
    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise OllamaError(f"响应结构异常：{data}")
    return content.strip()

