from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from config.web_search_config import WebSearchConfig
from shared.models import JSONValue, ToolRequest, ToolResult
from tools.http_client import HttpClient, HttpConfig, HttpResult

logger = logging.getLogger("SlavikAI.WebSearchTool")

SERPER_ENDPOINT = "https://google.serper.dev/search"


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
        self.config = config or WebSearchConfig(api_key=os.getenv("SERPER_API_KEY"))
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
        result: HttpResult = self.http.post_json(
            SERPER_ENDPOINT, json=payload, headers=headers, timeout=self.config.timeout
        )
        if not result.ok:
            return ToolResult.failure(f"HTTP ошибка поиска: {result.error}")
        if not isinstance(result.data, dict):
            return ToolResult.failure("Неверный формат ответа поиска.")

        results_raw = result.data.get("organic") or []
        ranked: list[SearchResult] = []
        for idx, item in enumerate(results_raw[: self.config.top_k]):
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
            meta={"provider": "serper"},
        )

    def _resolve_api_key(self) -> str | None:
        if self.config.api_key:
            return self.config.api_key
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
