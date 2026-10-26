from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config.memory_config import MemoryConfig
from config.tools_config import DEFAULT_TOOLS_STATE, ToolsConfig
from config.ui_embeddings_settings import UIEmbeddingsSettings
from config.web_search_config import WebSearchConfig
from core import agent_tools
from llm.types import ModelConfig
from memory import session_summarizer
from server.http.common import chat_payload, ui_settings
from server.http.common.model_config_snapshot import build_model_config_snapshot
from shared.models import JSONValue


@dataclass(frozen=True)
class SettingsRuntimeBindings:
    ui_settings_path_getter: Callable[[], Path]
    api_keys_path_getter: Callable[[], Path]
    model_config_path_getter: Callable[[], Path]
    load_memory_config_fn: Callable[[], MemoryConfig]
    save_memory_config_fn: Callable[[MemoryConfig], None]
    load_tools_config_fn: Callable[[], ToolsConfig]
    save_tools_config_fn: Callable[[ToolsConfig], None]

    def _ui_settings_path(self) -> Path:
        return self.ui_settings_path_getter()

    def _api_keys_path(self) -> Path:
        return self.api_keys_path_getter()

    def _model_config_path(self) -> Path:
        return self.model_config_path_getter()

    def provider_auth_headers(self, provider: str) -> tuple[dict[str, str], str | None]:
        return ui_settings._provider_auth_headers(
            provider,
            api_keys_path=self._api_keys_path(),
        )

    def fetch_provider_models(
        self,
        provider: str,
        *,
        allow_local_fallback: bool = True,
        discovery_only: bool = False,
    ) -> tuple[list[str], str | None]:
        return ui_settings._fetch_provider_models(
            provider,
            api_keys_path=self._api_keys_path(),
            allow_local_fallback=allow_local_fallback,
            discovery_only=discovery_only,
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

    def load_embeddings_settings(self) -> UIEmbeddingsSettings:
        return ui_settings._load_embeddings_settings(ui_settings_path=self._ui_settings_path())

    def save_embeddings_settings(self, settings: UIEmbeddingsSettings) -> None:
        ui_settings._save_embeddings_settings(settings, ui_settings_path=self._ui_settings_path())

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
        return ui_settings._load_provider_api_keys(api_keys_path=self._api_keys_path())

    def save_provider_api_keys(self, api_keys: dict[str, str]) -> None:
        ui_settings._save_provider_api_keys(api_keys, api_keys_path=self._api_keys_path())

    def drop_legacy_provider_api_keys(self) -> bool:
        return ui_settings._drop_legacy_provider_api_keys(ui_settings_path=self._ui_settings_path())

    def load_provider_runtime_checks(self) -> dict[str, dict[str, JSONValue]]:
        return ui_settings._load_provider_runtime_checks(ui_settings_path=self._ui_settings_path())

    def save_provider_runtime_checks(
        self,
        checks: dict[str, dict[str, JSONValue]],
    ) -> None:
        ui_settings._save_provider_runtime_checks(checks, ui_settings_path=self._ui_settings_path())

    def validate_provider_api_key(self, provider: str, api_key: str) -> str | None:
        return ui_settings._validate_provider_api_key(provider, api_key)

    def resolve_provider_api_key(
        self,
        provider: str,
        *,
        settings_api_keys: dict[str, str] | None = None,
    ) -> str | None:
        return ui_settings._resolve_provider_api_key(
            provider,
            settings_api_keys=settings_api_keys,
            api_keys_path=self._api_keys_path(),
        )

    def provider_api_key_source(
        self,
        provider: str,
        *,
        settings_api_keys: dict[str, str] | None = None,
    ) -> Literal["env", "file", "missing"]:
        return ui_settings._provider_api_key_source(
            provider,
            settings_api_keys=settings_api_keys,
            api_keys_path=self._api_keys_path(),
        )

    def provider_settings_payload(
        self, *, include_custom_providers: bool = True
    ) -> list[dict[str, JSONValue]]:
        return ui_settings._provider_settings_payload(
            ui_settings_path=self._ui_settings_path(),
            api_keys_path=self._api_keys_path(),
            include_custom_providers=include_custom_providers,
        )

    def load_default_model(self) -> ModelConfig | None:
        return ui_settings._load_default_model(model_config_path=self._model_config_path())

    def save_default_model(self, config: ModelConfig) -> None:
        ui_settings._save_default_model(config, model_config_path=self._model_config_path())

    def build_settings_payload(
        self,
        *,
        include_custom_providers: bool = True,
        include_context_budget: bool = False,
        include_model_config: bool = False,
        include_attachment_limits: bool = False,
        include_runtime_context_limits: bool = False,
        include_search_backend_defaults: bool = False,
    ) -> dict[str, JSONValue]:
        search_defaults = WebSearchConfig() if include_search_backend_defaults else None
        policy_profile, yolo_armed, yolo_armed_at = self.load_policy_settings()
        memory_config = self.load_memory_config_fn()
        embeddings_settings = self.load_embeddings_settings()
        tools_state = self.load_tools_state()
        tools_registry = {key: value for key, value in tools_state.items() if key != "safe_mode"}
        default_model = self.load_default_model()
        default_model_payload: dict[str, JSONValue] | None = None
        if default_model is not None:
            default_model_payload = {
                "provider": default_model.provider,
                "model": default_model.model,
            }
        if not include_custom_providers and default_model_payload is not None:
            provider_raw = default_model_payload.get("provider")
            if isinstance(provider_raw, str) and provider_raw.startswith(
                ui_settings.CUSTOM_PROVIDER_PREFIX
            ):
                default_model_payload = None
        return {
            "settings": {
                "model": default_model_payload,
                **(
                    {"model_config_snapshot": build_model_config_snapshot(default_model)}
                    if include_model_config
                    else {}
                ),
                **(
                    {
                        "context_budget_snapshot": {
                            "values": {
                                key: value
                                for key, value in memory_config.context_budget.to_dict().items()
                            },
                            "basis": "server_configuration_at_refresh",
                            "source": "Memory configuration with server defaults",
                        }
                    }
                    if include_context_budget
                    else {}
                ),
                **(
                    {
                        "attachment_limits_snapshot": {
                            "basis": "server_request_validation_at_refresh",
                            "values": {
                                "max_items": chat_payload.MAX_ATTACHMENTS_PER_MESSAGE,
                                "max_item_code_points": chat_payload.MAX_ATTACHMENT_CHARS,
                                "max_total_code_points": chat_payload.MAX_TOTAL_ATTACHMENTS_CHARS,
                            },
                        }
                    }
                    if include_attachment_limits
                    else {}
                ),
                **(
                    {
                        "runtime_context_limits_snapshot": {
                            "basis": "runtime_context_constants_at_refresh",
                            "values": {
                                "short_term_max_messages": agent_tools.MAX_SHORT_TERM_MESSAGES,
                                "summary_max_messages": session_summarizer.MAX_SUMMARY_MESSAGES,
                                "summary_message_code_points": (
                                    session_summarizer.MAX_TRANSCRIPT_MESSAGE_CHARS
                                ),
                            },
                        }
                    }
                    if include_runtime_context_limits
                    else {}
                ),
                **(
                    {
                        "search_backend_defaults_snapshot": {
                            "basis": "backend_class_defaults_at_refresh",
                            "values": {
                                "top_k": search_defaults.top_k,
                                "timeout_seconds": search_defaults.timeout,
                                "response_content_bytes": search_defaults.max_bytes,
                            },
                        }
                    }
                    if search_defaults is not None
                    else {}
                ),
                "memory": {
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
                "audio": {
                    "tts": ui_settings._tts_settings_payload(api_keys_path=self._api_keys_path()),
                },
                "providers": self.provider_settings_payload(
                    include_custom_providers=include_custom_providers
                ),
            },
        }
