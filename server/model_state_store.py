from __future__ import annotations

import asyncio
from pathlib import Path

from config.model_store import MODEL_CONFIG_PATH, load_model_configs, save_model_configs
from llm.types import ModelConfig


class ModelStateStore:
    def __init__(self, main: ModelConfig | None = None) -> None:
        self._main = main
        self._lock = asyncio.Lock()

    async def get_main(self) -> ModelConfig | None:
        async with self._lock:
            return self._main

    async def set_main(self, main: ModelConfig | None) -> None:
        async with self._lock:
            self._main = main


class FileBackedModelStateStore(ModelStateStore):
    def __init__(self, path: Path = MODEL_CONFIG_PATH) -> None:
        self._path = path
        super().__init__(load_model_configs(path))

    async def set_main(self, main: ModelConfig | None) -> None:
        async with self._lock:
            self._main = main
            save_model_configs(main, self._path)
