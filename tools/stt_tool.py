from __future__ import annotations

from pathlib import Path

from config.stt_config import SttConfig
from shared.models import JSONValue, ToolRequest, ToolResult
from tools.http_client import HttpClient, HttpResult

SANDBOX_AUDIO = Path("sandbox/audio")
SANDBOX_AUDIO.mkdir(parents=True, exist_ok=True)


class SttTool:
    def __init__(self, http_client: HttpClient, config: SttConfig | None = None) -> None:
        self.http = http_client
        self.config = config or SttConfig()

    def handle(self, request: ToolRequest) -> ToolResult:
        file_path_raw = str(request.args.get("file_path") or "").strip()
        language = str(request.args.get("language") or self.config.language)
        if not file_path_raw:
            return ToolResult.failure("Не указан путь к аудио.")

        file_path = (SANDBOX_AUDIO / Path(file_path_raw).name).resolve()
        if not str(file_path).startswith(str(SANDBOX_AUDIO.resolve())):
            return ToolResult.failure("Файл вне sandbox/audio.")
        if not file_path.exists():
            return ToolResult.failure("Файл не найден.")

        api_key = self.config.resolve_api_key()
        if not api_key:
            return ToolResult.failure("STT API key не задан (STT_API_KEY).")

        files = {"file": file_path.open("rb")}
        data = {"language": language}

        try:
            result: HttpResult = self.http._request(
                "POST",
                self.config.endpoint,
                expect_json=True,
                as_bytes=False,
                files=files,
                data=data,
                timeout=self.config.timeout,
            )
        finally:
            files["file"].close()

        if not result.ok or not isinstance(result.data, dict):
            return ToolResult.failure(result.error or "Ошибка STT сервиса.")

        text = str(result.data.get("text") or "").strip()
        if not text:
            return ToolResult.failure("STT вернул пустой текст.")
        confidence_raw = result.data.get("confidence")
        meta: dict[str, JSONValue] = {"language": language}
        if isinstance(confidence_raw, (int, float)):
            meta["confidence"] = float(confidence_raw)
        confidence = meta.get("confidence")
        return ToolResult.success(
            {"output": text, "text": text, "language": language, "confidence": confidence},
            meta=meta,
        )
