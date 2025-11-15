from __future__ import annotations

"""
Small helper script to debug the privateGPT-API proxy.

Usage (PowerShell):
    python tools/debug_proxy.py --base-url http://localhost:8000/v1 \
        --api-key dummy --model gpt-4o-mini

If OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL are already set in the
environment, you can omit the flags.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ProxyConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(
        cls,
        *,
        base_url: Optional[str],
        api_key: Optional[str],
        model: Optional[str],
    ) -> "ProxyConfig":
        resolved_base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:8000/v1"
        ).rstrip("/")
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or "debug-key"
        resolved_model = model or os.getenv("OPENAI_MODEL") or "azure-gpt-4o"
        return cls(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            model=resolved_model,
        )

    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def debug_embeddings(cfg: ProxyConfig) -> None:
    payload = {
        "model": "text-embedding-3-small",
        "input": ["Xin chào", "DemoLangGraph test"],
    }
    url = f"{cfg.base_url}/embeddings"
    print(f"\n[Embeddings] POST {url}")
    embeddings = _post_json(url, cfg.headers(), payload)
    print(json.dumps(embeddings, indent=2, ensure_ascii=False))


def debug_completion(cfg: ProxyConfig) -> None:
    payload = {
        "model": cfg.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Tóm tắt ngắn gọn quy trình ingest của DemoLangGraph.",
            },
        ],
    }
    url = f"{cfg.base_url}/chat/completions"
    print(f"\n[Chat] POST {url}")
    completion = _post_json(url, cfg.headers(), payload)
    print(json.dumps(completion, indent=2, ensure_ascii=False))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug privateGPT-API proxy.")
    parser.add_argument("--base-url", dest="base_url", type=str, default=None)
    parser.add_argument("--api-key", dest="api_key", type=str, default=None)
    parser.add_argument("--model", dest="model", type=str, default=None)
    parser.add_argument(
        "--skip-embeddings", action="store_true", help="Skip embedding check."
    )
    parser.add_argument(
        "--skip-chat", action="store_true", help="Skip chat completion check."
    )
    args = parser.parse_args()

    cfg = ProxyConfig.from_env(
        base_url=args.base_url, api_key=args.api_key, model=args.model
    )

    print("Using proxy config:")
    print(f"  base_url = {cfg.base_url}")
    print(f"  model    = {cfg.model}")

    try:
        if not args.skip_embeddings:
            debug_embeddings(cfg)
        if not args.skip_chat:
            debug_completion(cfg)
    except requests.HTTPError as exc:
        print("\nRequest failed:")
        print(f"  Status: {exc.response.status_code}")
        print(f"  URL: {exc.request.url}")
        print(f"  Body: {exc.response.text[:500]}")
        raise


if __name__ == "__main__":
    main()
