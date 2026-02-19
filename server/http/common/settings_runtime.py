from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config.memory_config import MemoryConfig
from config.tools_config import DEFAULT_TOOLS_STATE, ToolsConfig
from config.ui_embeddings_settings import UIEmbeddingsSettings
from server.http.common import ui_settings
from shared.models import JSONValue


@dataclass(frozen=True)
class SettingsRuntimeBindings:
    ui_settings_path_getter: Callable[[], Path]
    load_memory_config_fn: Callable[[], MemoryConfig]
    save_memory_config_fn: Callable[[MemoryConfig], None]
    load_tools_config_fn: Callable[[], ToolsConfig]
    save_tools_config_fn: Callable[[ToolsConfig], None]

    def _ui_settings_path(self) -> Path:
        return self.ui_settings_path_getter()

    def provider_auth_headers(self, provider: str) -> tuple[dict[str, str], str | None]:
        return ui_settings._provider_auth_headers(
            provider,
            ui_settings_path=self._ui_settings_path(),
        )

    def fetch_provider_models(self, provider: str) -> tuple[list[str], str | None]:
        return ui_settings._fetch_provider_models(
            provider,
            ui_settings_path=self._ui_settings_path(),
        )

    def load_memory_config_runtime(self) -> MemoryConfig:
        return self.load_memory_config_fn()

    def save_memory_config_runtime(self, config: MemoryConfig) -> None:
        self.save_memory_config_fn(config)

    def load_tools_state(self) -> dict[str, bool]:
        try:
            return self.load_tools_config_fn().to_dict()
        except Exception:  # noqa: BLE001
            return dict(DEFAULT_TOOLS_STATE)

    def save_tools_state(self, state: dict[str, bool]) -> None:
        payload: dict[str, object] = {key: value for key, value in state.items()}
        self.save_tools_config_fn(ToolsConfig.from_dict(payload))

    def load_ui_settings_blob(self) -> dict[str, object]:
        return ui_settings._load_ui_settings_blob(ui_settings_path=self._ui_settings_path())

    def save_ui_settings_blob(self, payload: dict[str, object]) -> None:
        ui_settings._save_ui_settings_blob(payload, ui_settings_path=self._ui_settings_path())

    def load_personalization_settings(self) -> tuple[str, str]:
        return ui_settings._load_personalization_settings(ui_settings_path=self._ui_settings_path())

    def save_personalization_settings(self, *, tone: str, system_prompt: str) -> None:
        ui_settings._save_personalization_settings(
            tone=tone,
            system_prompt=system_prompt,
            ui_settings_path=self._ui_settings_path(),
        )

    def load_embeddings_settings(self) -> UIEmbeddingsSettings:
        return ui_settings._load_embeddings_settings(ui_settings_path=self._ui_settings_path())

    def save_embeddings_settings(self, settings: UIEmbeddingsSettings) -> None:
        ui_settings._save_embeddings_settings(settings, ui_settings_path=self._ui_settings_path())

    def load_composer_settings(self) -> tuple[bool, int]:
        return ui_settings._load_composer_settings(ui_settings_path=self._ui_settings_path())

    def save_composer_settings(
        self,
        *,
        long_paste_to_file_enabled: bool,
        long_paste_threshold_chars: int,
    ) -> None:
        ui_settings._save_composer_settings(
            long_paste_to_file_enabled=long_paste_to_file_enabled,
            long_paste_threshold_chars=long_paste_threshold_chars,
            ui_settings_path=self._ui_settings_path(),
        )

    def load_policy_settings(self) -> tuple[str, bool, str | None]:
        return ui_settings._load_policy_settings(ui_settings_path=self._ui_settings_path())

    def save_policy_settings(
        self,
        *,
        profile: str,
        yolo_armed: bool,
        yolo_armed_at: str | None,
    ) -> None:
        ui_settings._save_policy_settings(
            profile=profile,
            yolo_armed=yolo_armed,
            yolo_armed_at=yolo_armed_at,
            ui_settings_path=self._ui_settings_path(),
        )

    def load_provider_api_keys(self) -> dict[str, str]:
        return ui_settings._load_provider_api_keys(ui_settings_path=self._ui_settings_path())

    def save_provider_api_keys(self, api_keys: dict[str, str]) -> None:
        ui_settings._save_provider_api_keys(api_keys, ui_settings_path=self._ui_settings_path())

    def load_provider_runtime_checks(self) -> dict[str, dict[str, JSONValue]]:
        return ui_settings._load_provider_runtime_checks(ui_settings_path=self._ui_settings_path())

    def save_provider_runtime_checks(
        self,
        checks: dict[str, dict[str, JSONValue]],
    ) -> None:
        ui_settings._save_provider_runtime_checks(checks, ui_settings_path=self._ui_settings_path())

    def resolve_provider_api_key(
        self,
        provider: str,
        *,
        settings_api_keys: dict[str, str] | None = None,
    ) -> str | None:
        return ui_settings._resolve_provider_api_key(
            provider,
            settings_api_keys=settings_api_keys,
            ui_settings_path=self._ui_settings_path(),
        )

    def provider_api_key_source(
        self,
        provider: str,
        *,
        settings_api_keys: dict[str, str] | None = None,
    ) -> Literal["settings", "env", "missing"]:
        return ui_settings._provider_api_key_source(
            provider,
            settings_api_keys=settings_api_keys,
            ui_settings_path=self._ui_settings_path(),
        )

    def provider_settings_payload(self) -> list[dict[str, JSONValue]]:
        return ui_settings._provider_settings_payload(ui_settings_path=self._ui_settings_path())

    def build_settings_payload(self) -> dict[str, JSONValue]:
        tone, system_prompt = self.load_personalization_settings()
        long_paste_to_file_enabled, long_paste_threshold_chars = self.load_composer_settings()
        policy_profile, yolo_armed, yolo_armed_at = self.load_policy_settings()
        memory_config = self.load_memory_config_fn()
        embeddings_settings = self.load_embeddings_settings()
        tools_state = self.load_tools_state()
        tools_registry = {key: value for key, value in tools_state.items() if key != "safe_mode"}
        return {
            "settings": {
                "personalization": {"tone": tone, "system_prompt": system_prompt},
                "composer": {
                    "long_paste_to_file_enabled": long_paste_to_file_enabled,
                    "long_paste_threshold_chars": long_paste_threshold_chars,
                },
                "memory": {
                    "auto_save_dialogue": memory_config.auto_save_dialogue,
                    "inbox_max_items": memory_config.inbox_max_items,
                    "inbox_ttl_days": memory_config.inbox_ttl_days,
                    "inbox_writes_per_minute": memory_config.inbox_writes_per_minute,
                    "embeddings": {
                        "provider": embeddings_settings.provider,
                        "local_model": embeddings_settings.local_model,
                        "openai_model": embeddings_settings.openai_model,
                    },
                },
                "tools": {
                    "state": tools_state,
                    "registry": tools_registry,
                },
                "policy": {
                    "profile": policy_profile,
                    "yolo_armed": yolo_armed,
                    "yolo_armed_at": yolo_armed_at,
                },
                "providers": self.provider_settings_payload(),
            },
        }
