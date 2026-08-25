from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlencode, urlparse

from config.web_search_config import WebSearchConfig
from shared.models import JSONValue, ToolRequest, ToolResult
from tools.http_client import HttpClient, HttpConfig, HttpResult

logger = logging.getLogger("SlavikAI.WebSearchTool")

SERPER_ENDPOINT = "https://google.serper.dev/search"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0


class WebSearchTool:
    def __init__(
        self,
        config: WebSearchConfig | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            provider_raw = os.getenv("WEB_SEARCH_PROVIDER", "serper").strip().lower()
            provider: Literal["serper", "serpapi"] = cast(
                "Literal['serper', 'serpapi']",
                provider_raw if provider_raw in {"serper", "serpapi"} else "serper",
            )
            key_env = "SERPAPI_API_KEY" if provider == "serpapi" else "SERPER_API_KEY"
            self.config = WebSearchConfig(provider=provider, api_key=os.getenv(key_env))
        self.http = http_client or HttpClient(
            HttpConfig(timeout=self.config.timeout, max_bytes=self.config.max_bytes)
        )

    def handle(self, request: ToolRequest) -> ToolResult:
        query_raw = str(request.args.get("query") or "").strip()
        if not query_raw:
            return ToolResult.failure("Запрос пуст.")

        parsed = urlparse(query_raw)
        if parsed.scheme and parsed.scheme not in ("http", "https"):
            return ToolResult.failure("Неверный URL. Разрешены только http/https.")

        # Если передан URL, просто скачиваем содержимое
        if query_raw.startswith(("http://", "https://")):
            return self._fetch_url(query_raw)

        if self.config.provider == "serper":
            return self._search_serper(query_raw)
        if self.config.provider == "serpapi":
            return self._search_serpapi(query_raw)
        return ToolResult.failure(f"Неизвестный провайдер поиска: {self.config.provider}")

    def _fetch_url(self, url: str) -> ToolResult:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return ToolResult.failure("Неверный URL. Разрешены только http/https.")
        result = self.http.get_text(url)
        if not result.ok or not isinstance(result.data, str):
            error = result.error or "Ошибка загрузки страницы."
            return ToolResult.failure(f"HTTP ошибка: {error}", {"status": result.status_code})
        return ToolResult.success(
            {
                "output": result.data,
                "status": result.status_code or 0,
                "url": url,
            },
            meta={"truncated": (result.meta or {}).get("truncated", False)},
        )

    def _search_serper(self, query: str) -> ToolResult:
        api_key = self._resolve_api_key()
        if not api_key:
            return ToolResult.failure("SERPER_API_KEY не задан. Установите ключ для web поиска.")

        payload = {"q": query, "num": self.config.top_k}
        headers = {"X-API-KEY": api_key}
        result: HttpResult = self._request_with_retry(
            lambda: self.http.post_json(
                SERPER_ENDPOINT, json=payload, headers=headers, timeout=self.config.timeout
            )
        )
        if not result.ok:
            if result.status_code in (401, 403):
                return ToolResult.failure(
                    f"HTTP ошибка поиска: {result.error} — ключ SERPER_API_KEY недействителен "
                    "или нет доступа. Проверьте ключ в Settings/окружении."
                )
            return ToolResult.failure(f"HTTP ошибка поиска: {result.error}")
        if not isinstance(result.data, dict):
            return ToolResult.failure("Неверный формат ответа поиска.")

        return self._build_results_payload(result.data, provider="serper")

    def _search_serpapi(self, query: str) -> ToolResult:
        api_key = self._resolve_api_key()
        if not api_key:
            return ToolResult.failure("SERPAPI_API_KEY не задан. Установите ключ для web поиска.")

        params = {
            "engine": "google",
            "q": query,
            "num": self.config.top_k,
            "api_key": api_key,
        }
        url = f"{SERPAPI_ENDPOINT}?{urlencode(params)}"
        result = self._request_with_retry(
            lambda: self.http.get_text(url, timeout=self.config.timeout)
        )
        if not result.ok or not isinstance(result.data, str):
            error = result.error or "Ошибка поиска."
            return ToolResult.failure(
                f"HTTP ошибка поиска: {error}",
                {"status": result.status_code},
            )
        try:
            data = json.loads(result.data)
        except (ValueError, TypeError):
            return ToolResult.failure("Неверный формат ответа поиска.")
        if not isinstance(data, dict):
            return ToolResult.failure("Неверный формат ответа поиска.")
        return self._build_results_payload(data, provider="serpapi")

    def _request_with_retry(self, request_fn: Callable[[], HttpResult]) -> HttpResult:
        """Один повтор на timeout/5xx: внешние поиски иногда отвечают медленно."""
        last: HttpResult | None = None
        for attempt in range(2):
            result = request_fn()
            if result.ok:
                return result
            status_code = result.status_code
            retryable = status_code is None or (status_code or 0) >= 500
            if attempt == 0 and retryable:
                time.sleep(1.0)
                last = result
                continue
            return result
        return (
            last
            if last is not None
            else HttpResult(
                ok=False,
                data=None,
                status_code=None,
                error="timeout",
                headers={},
                meta={},
            )
        )

    def _build_results_payload(
        self,
        data: dict[str, object],
        *,
        provider: str,
    ) -> ToolResult:
        raw_results = data.get("organic_results") or data.get("organic")
        results_raw: list[object] = raw_results if isinstance(raw_results, list) else []
        ranked: list[SearchResult] = []
        for idx, item in enumerate(results_raw[: self.config.top_k]):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            link = str(item.get("link") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            if not title or not link:
                continue
            ranked.append(self._score_result(title, link, snippet, idx))

        if not ranked:
            return ToolResult.failure("Результатов не найдено.")

        ranked_sorted = sorted(ranked, key=lambda r: r.score, reverse=True)
        serialized: list[dict[str, JSONValue]] = [
            {"title": r.title, "url": r.url, "snippet": r.snippet, "score": round(r.score, 4)}
            for r in ranked_sorted
        ]

        return ToolResult.success(
            {
                "output": "\n".join(
                    [
                        f"{str(r['title'])} — {str(r['url'])}\n{str(r['snippet'])}"
                        for r in serialized
                    ]
                ),
                "results": serialized,
            },
            meta={"provider": provider},
        )

    def _resolve_api_key(self) -> str | None:
        if self.config.api_key:
            return self.config.api_key
        key_env = "SERPAPI_API_KEY" if self.config.provider == "serpapi" else "SERPER_API_KEY"
        env_key = os.getenv(key_env)
        if env_key:
            return env_key
        # Совместимость: ключ мог быть положен в старую переменную.
        env_key = os.getenv("SERPER_API_KEY")
        if env_key:
            return env_key
        key_path = Path("config/web_search_api_key.txt")
        if key_path.exists():
            return key_path.read_text(encoding="utf-8").strip()
        return None

    def _score_result(self, title: str, url: str, snippet: str, position: int) -> SearchResult:
        # детерминированный скоринг: позиция + насыщенность сниппета
        richness = min(len(snippet) / 400.0, 1.0)
        positional_bonus = max(0.0, 1.0 - position * 0.05)
        score = round(positional_bonus + richness, 6)
        return SearchResult(title=title, url=url, snippet=snippet, score=score)
