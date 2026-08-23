from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, cast

from aiohttp import web
from aiohttp.multipart import BodyPartReader

from config.memory_config import MemoryConfig
from config.ui_embeddings_settings import UIEmbeddingsSettings
from server import http_api as api
from server.agent_provider import ScopedAgentProvider
from server.http.common.responses import error_response, json_response
from server.http.common.runtime_contract import RuntimeModelStateProtocol
from shared.models import JSONValue

logger = logging.getLogger("SlavikAI.HttpAPI")
_SANDBOX_AUDIO_ROOT = Path("sandbox/audio").resolve()


def _openai_error_message(response: object) -> str | None:
    if not hasattr(response, "json"):
        return None
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    error_raw = payload.get("error")
    if not isinstance(error_raw, dict):
        return None
    message_raw = error_raw.get("message")
    if not isinstance(message_raw, str):
        return None
    normalized = message_raw.strip()
    return normalized or None


def _tts_error_response(message: str) -> web.Response:
    normalized = message.strip() or "OpenAI TTS request failed."
    lowered = normalized.lower()
    status = 502
    error_type = "upstream_error"
    code = "tts_failed"
    if (
        "пуст" in lowered
        or "format" in lowered
        or "формат" in lowered
        or "voice" in lowered
        or "превышает лимит" in lowered
        or "4096" in lowered
    ):
        status = 400
        error_type = "invalid_request_error"
        code = "invalid_request_error"
    elif "api key" in lowered or "отключ" in lowered or "safe mode" in lowered:
        status = 409
        error_type = "configuration_error"
        code = "tts_unavailable"
    return error_response(
        status=status,
        message=normalized,
        error_type=error_type,
        code=code,
    )


async def handle_ui_settings(request: web.Request) -> web.Response:
    del request
    try:
        payload = api._build_settings_payload()
        logger.info(
            "Settings payload prepared",
            extra={
                "path": "/ui/api/settings",
                "reason": "success",
                "settings_snapshot": payload,
            },
        )
        return json_response(payload)
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=500,
            message=f"Не удалось загрузить settings: {exc}",
            error_type="internal_error",
            code="settings_load_failed",
        )


async def handle_ui_settings_update(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    forbidden_key = api._user_plane_forbidden_settings_key(payload)
    if forbidden_key is not None:
        return error_response(
            status=403,
            message=f"Control-plane поле запрещено в user-plane: {forbidden_key}",
            error_type="forbidden",
            code="security_fields_forbidden",
        )
    try:
        api._drop_legacy_provider_api_keys()
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=500,
            message=f"Не удалось очистить legacy providers API keys: {exc}",
            error_type="internal_error",
            code="settings_update_failed",
        )

    next_embeddings_settings: UIEmbeddingsSettings | None = None

    appearance_raw = payload.get("appearance")
    if appearance_raw is not None:
        if not isinstance(appearance_raw, dict):
            return error_response(
                status=400,
                message="appearance должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        theme_raw = appearance_raw.get("theme")
        if theme_raw is not None:
            if not isinstance(theme_raw, str):
                return error_response(
                    status=400,
                    message="appearance.theme должен быть строкой.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            theme = theme_raw.strip().lower()
            if theme not in api.APPEARANCE_THEMES:
                return error_response(
                    status=400,
                    message="appearance.theme должен быть default или oled.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            api._save_appearance_settings(theme=theme)

    personalization_raw = payload.get("personalization")
    if personalization_raw is not None:
        if not isinstance(personalization_raw, dict):
            return error_response(
                status=400,
                message="personalization должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        tone, system_prompt = api._load_personalization_settings()
        if "tone" in personalization_raw:
            tone_raw = personalization_raw.get("tone")
            if not isinstance(tone_raw, str):
                return error_response(
                    status=400,
                    message="personalization.tone должен быть строкой.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            normalized_tone = tone_raw.strip()
            if not normalized_tone:
                return error_response(
                    status=400,
                    message="personalization.tone не должен быть пустым.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            tone = normalized_tone
        if "system_prompt" in personalization_raw:
            prompt_raw = personalization_raw.get("system_prompt")
            if not isinstance(prompt_raw, str):
                return error_response(
                    status=400,
                    message="personalization.system_prompt должен быть строкой.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            system_prompt = prompt_raw
        api._save_personalization_settings(tone=tone, system_prompt=system_prompt)

    composer_raw = payload.get("composer")
    if composer_raw is not None:
        if not isinstance(composer_raw, dict):
            return error_response(
                status=400,
                message="composer должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        long_paste_to_file_enabled, long_paste_threshold_chars = api._load_composer_settings()
        if "long_paste_to_file_enabled" in composer_raw:
            enabled_raw = composer_raw.get("long_paste_to_file_enabled")
            if not isinstance(enabled_raw, bool):
                return error_response(
                    status=400,
                    message="composer.long_paste_to_file_enabled должен быть bool.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            long_paste_to_file_enabled = enabled_raw
        if "long_paste_threshold_chars" in composer_raw:
            threshold_raw = composer_raw.get("long_paste_threshold_chars")
            if isinstance(threshold_raw, bool) or not isinstance(threshold_raw, int):
                return error_response(
                    status=400,
                    message="composer.long_paste_threshold_chars должен быть int.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            if (
                threshold_raw < api.MIN_LONG_PASTE_THRESHOLD_CHARS
                or threshold_raw > api.MAX_LONG_PASTE_THRESHOLD_CHARS
            ):
                return error_response(
                    status=400,
                    message=(
                        "composer.long_paste_threshold_chars вне диапазона "
                        f"{api.MIN_LONG_PASTE_THRESHOLD_CHARS}..{api.MAX_LONG_PASTE_THRESHOLD_CHARS}."
                    ),
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            long_paste_threshold_chars = threshold_raw
        api._save_composer_settings(
            long_paste_to_file_enabled=long_paste_to_file_enabled,
            long_paste_threshold_chars=long_paste_threshold_chars,
        )

    memory_raw = payload.get("memory")
    if memory_raw is not None:
        if not isinstance(memory_raw, dict):
            return error_response(
                status=400,
                message="memory должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        current_memory = api._load_memory_config_runtime()
        inbox_max_items = current_memory.inbox_max_items
        inbox_ttl_days = current_memory.inbox_ttl_days
        inbox_writes_per_minute = current_memory.inbox_writes_per_minute
        embeddings_settings = api._load_embeddings_settings()
        if "auto_save_dialogue" in memory_raw:
            return error_response(
                status=400,
                message=(
                    "memory.auto_save_dialogue больше не поддерживается: "
                    "запись Memory требует отдельного подтверждения."
                ),
                error_type="invalid_request_error",
                code="memory_confirmation_required",
            )
        if "inbox_max_items" in memory_raw:
            raw_value = memory_raw.get("inbox_max_items")
            if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
                return error_response(
                    status=400,
                    message="memory.inbox_max_items должен быть положительным int.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            inbox_max_items = raw_value
        if "inbox_ttl_days" in memory_raw:
            raw_value = memory_raw.get("inbox_ttl_days")
            if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
                return error_response(
                    status=400,
                    message="memory.inbox_ttl_days должен быть положительным int.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            inbox_ttl_days = raw_value
        if "inbox_writes_per_minute" in memory_raw:
            raw_value = memory_raw.get("inbox_writes_per_minute")
            if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
                return error_response(
                    status=400,
                    message="memory.inbox_writes_per_minute должен быть положительным int.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            inbox_writes_per_minute = raw_value
        if "embeddings" in memory_raw:
            embeddings_raw = memory_raw.get("embeddings")
            if not isinstance(embeddings_raw, dict):
                return error_response(
                    status=400,
                    message="memory.embeddings должен быть объектом.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            provider = embeddings_settings.provider
            local_model = embeddings_settings.local_model
            openai_model = embeddings_settings.openai_model

            if "provider" in embeddings_raw:
                provider_raw = embeddings_raw.get("provider")
                if not isinstance(provider_raw, str):
                    return error_response(
                        status=400,
                        message="memory.embeddings.provider должен быть строкой.",
                        error_type="invalid_request_error",
                        code="invalid_request_error",
                    )
                normalized_provider = provider_raw.strip().lower()
                if normalized_provider not in {"local", "openai"}:
                    return error_response(
                        status=400,
                        message="memory.embeddings.provider должен быть local|openai.",
                        error_type="invalid_request_error",
                        code="invalid_request_error",
                    )
                provider = cast(Literal["local", "openai"], normalized_provider)

            if "local_model" in embeddings_raw:
                local_model_raw = embeddings_raw.get("local_model")
                if not isinstance(local_model_raw, str):
                    return error_response(
                        status=400,
                        message="memory.embeddings.local_model должен быть строкой.",
                        error_type="invalid_request_error",
                        code="invalid_request_error",
                    )
                normalized_local = local_model_raw.strip()
                if not normalized_local:
                    return error_response(
                        status=400,
                        message="memory.embeddings.local_model не должен быть пустым.",
                        error_type="invalid_request_error",
                        code="invalid_request_error",
                    )
                local_model = normalized_local

            if "openai_model" in embeddings_raw:
                openai_model_raw = embeddings_raw.get("openai_model")
                if not isinstance(openai_model_raw, str):
                    return error_response(
                        status=400,
                        message="memory.embeddings.openai_model должен быть строкой.",
                        error_type="invalid_request_error",
                        code="invalid_request_error",
                    )
                normalized_openai = openai_model_raw.strip()
                if not normalized_openai:
                    return error_response(
                        status=400,
                        message="memory.embeddings.openai_model не должен быть пустым.",
                        error_type="invalid_request_error",
                        code="invalid_request_error",
                    )
                openai_model = normalized_openai

            embeddings_settings = UIEmbeddingsSettings(
                provider=provider,
                local_model=local_model,
                openai_model=openai_model,
            )
            api._save_embeddings_settings(embeddings_settings)
            next_embeddings_settings = embeddings_settings

        if "embeddings_model" in memory_raw:
            raw_model = memory_raw.get("embeddings_model")
            if not isinstance(raw_model, str):
                return error_response(
                    status=400,
                    message="memory.embeddings_model должен быть строкой.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            normalized_model = raw_model.strip()
            if not normalized_model:
                return error_response(
                    status=400,
                    message="memory.embeddings_model не должен быть пустым.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            embeddings_settings = UIEmbeddingsSettings(
                provider="local",
                local_model=normalized_model,
                openai_model=embeddings_settings.openai_model,
            )
            api._save_embeddings_settings(embeddings_settings)
            next_embeddings_settings = embeddings_settings
        api._save_memory_config_runtime(
            MemoryConfig(
                inbox_max_items=inbox_max_items,
                inbox_ttl_days=inbox_ttl_days,
                inbox_writes_per_minute=inbox_writes_per_minute,
            ),
        )

    providers_raw = payload.get("providers")
    if providers_raw is not None:
        if not isinstance(providers_raw, dict):
            return error_response(
                status=400,
                message="providers должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        try:
            next_api_keys = api._load_provider_api_keys()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                status=500,
                message=f"Не удалось прочитать API-ключи: {exc}",
                error_type="internal_error",
                code="api_key_store_error",
            )
        changed_providers: set[str] = set()
        for provider_name, provider_payload in providers_raw.items():
            if provider_name not in api.API_KEY_SETTINGS_PROVIDERS:
                return error_response(
                    status=400,
                    message=f"Неизвестный provider: {provider_name}",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            api_key_raw: object | None
            if provider_payload is None:
                api_key_raw = None
            elif isinstance(provider_payload, str):
                api_key_raw = provider_payload
            elif isinstance(provider_payload, dict):
                unexpected = sorted(str(key) for key in provider_payload if key != "api_key")
                if unexpected:
                    return error_response(
                        status=400,
                        message=(
                            f"providers.{provider_name} содержит неподдерживаемые поля: "
                            f"{', '.join(unexpected)}."
                        ),
                        error_type="invalid_request_error",
                        code="invalid_request_error",
                    )
                api_key_raw = provider_payload.get("api_key")
            else:
                return error_response(
                    status=400,
                    message=(f"providers.{provider_name} должен быть объектом, строкой или null."),
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            if api_key_raw is None:
                next_api_keys.pop(provider_name, None)
                changed_providers.add(provider_name)
                continue
            if not isinstance(api_key_raw, str):
                return error_response(
                    status=400,
                    message=(f"providers.{provider_name}.api_key должен быть строкой или null."),
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            normalized_key = api_key_raw.strip()
            if not normalized_key:
                next_api_keys.pop(provider_name, None)
                changed_providers.add(provider_name)
                continue
            validation_error = api._validate_provider_api_key(provider_name, normalized_key)
            if validation_error is not None:
                return error_response(
                    status=400,
                    message=validation_error,
                    error_type="invalid_request_error",
                    code="invalid_api_key",
                    details={"provider": provider_name},
                )
            next_api_keys[provider_name] = normalized_key
            changed_providers.add(provider_name)
        try:
            api._save_provider_api_keys(next_api_keys)
        except Exception as exc:  # noqa: BLE001
            return error_response(
                status=500,
                message=f"Не удалось сохранить API-ключи: {exc}",
                error_type="internal_error",
                code="api_key_store_error",
            )
        if changed_providers:
            next_runtime_checks = api._load_provider_runtime_checks()
            checked_at = api._utc_now_iso()
            for checked_provider in changed_providers:
                key_is_saved = bool(next_api_keys.get(checked_provider, "").strip())
                next_runtime_checks[checked_provider] = {
                    "api_key_valid": True if key_is_saved else None,
                    "last_check_error": None,
                    "last_checked_at": checked_at,
                }
            api._save_provider_runtime_checks(next_runtime_checks)

    model_raw = payload.get("model")
    if model_raw is not None:
        if not isinstance(model_raw, dict):
            return error_response(
                status=400,
                message="model должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        unexpected = sorted(str(key) for key in model_raw if key not in {"provider", "model"})
        if unexpected:
            return error_response(
                status=400,
                message=f"model содержит неподдерживаемые поля: {', '.join(unexpected)}.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        provider_raw = model_raw.get("provider")
        model_id_raw = model_raw.get("model")
        if not isinstance(provider_raw, str) or not isinstance(model_id_raw, str):
            return error_response(
                status=400,
                message="model.provider и model.model должны быть строками.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        model_provider = api._normalize_provider(provider_raw)
        model_id = model_id_raw.strip()
        if model_provider is None or not model_id:
            return error_response(
                status=400,
                message="Нужны корректные model.provider и model.model.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        persisted_model = api._load_default_model()
        model_changed = (
            persisted_model is None
            or persisted_model.provider != model_provider
            or persisted_model.model != model_id
        )
        if model_changed:
            models, fetch_error = api._fetch_provider_models(
                model_provider, allow_local_fallback=False
            )
            if fetch_error is not None:
                return error_response(
                    status=502,
                    message=fetch_error,
                    error_type="provider_error",
                    code="provider_models_unavailable",
                )
            if model_id not in models:
                return error_response(
                    status=404,
                    message=f"Модель '{model_id}' не найдена у провайдера '{model_provider}'.",
                    error_type="invalid_request_error",
                    code="model_not_found",
                )
            default_model = api._build_model_config(model_provider, model_id)
            api._save_default_model(default_model)
            runtime_model_state = cast(
                RuntimeModelStateProtocol, request.app["runtime_model_state"]
            )
            await runtime_model_state.set_global_main(default_model)
            agent_provider = cast(
                ScopedAgentProvider[api.AgentProtocol], request.app["agent_provider"]
            )

            def _apply_main_model(agent: api.AgentProtocol) -> None:
                agent.reconfigure_models(
                    default_model,
                    main_api_key=api._resolve_provider_api_key(model_provider),
                    persist=False,
                )

            await agent_provider.apply_to_existing(_apply_main_model)

    if next_embeddings_settings is not None:
        agent_provider = cast(ScopedAgentProvider[api.AgentProtocol], request.app["agent_provider"])

        def _apply_embeddings(agent: api.AgentProtocol) -> None:
            embeddings_model_for_agent = (
                next_embeddings_settings.local_model
                if next_embeddings_settings.provider == "local"
                else next_embeddings_settings.openai_model
            )
            set_embeddings_config = getattr(agent, "set_embeddings_config", None)
            set_embeddings_model = getattr(agent, "set_embeddings_model", None)
            if callable(set_embeddings_config):
                set_embeddings_config(
                    provider=next_embeddings_settings.provider,
                    local_model=next_embeddings_settings.local_model,
                    openai_model=next_embeddings_settings.openai_model,
                    openai_api_key=api._resolve_provider_api_key("openai"),
                )
            elif callable(set_embeddings_model):
                set_embeddings_model(embeddings_model_for_agent)

        await agent_provider.apply_to_existing(_apply_embeddings)

    return json_response(api._build_settings_payload())


async def handle_admin_security_settings_update(request: web.Request) -> web.Response:
    admin_error = api._require_admin_bearer(request)
    if admin_error is not None:
        return admin_error
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    invalid_keys = sorted(
        {
            str(key).strip()
            for key in payload
            if str(key).strip() not in api.UI_SETTINGS_ADMIN_ALLOWED_TOP_LEVEL_KEYS
        }
    )
    if invalid_keys:
        return error_response(
            status=400,
            message=f"Неподдерживаемые control-plane поля: {', '.join(invalid_keys)}.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not payload:
        return error_response(
            status=400,
            message="Нужно передать control-plane изменения (tools и/или policy).",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    next_policy_profile: str | None = None
    policy_raw = payload.get("policy")
    if policy_raw is not None:
        if not isinstance(policy_raw, dict):
            return error_response(
                status=400,
                message="policy должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        profile, yolo_armed, yolo_armed_at = api._load_policy_settings()
        if "profile" in policy_raw:
            profile = api._normalize_policy_profile(policy_raw.get("profile"))
        if "yolo_armed" in policy_raw:
            yolo_armed_raw = policy_raw.get("yolo_armed")
            if not isinstance(yolo_armed_raw, bool):
                return error_response(
                    status=400,
                    message="policy.yolo_armed должен быть bool.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            yolo_armed = yolo_armed_raw
        if profile == "yolo" or yolo_armed:
            confirm_raw = policy_raw.get("yolo_confirm")
            confirm_text_raw = policy_raw.get("yolo_confirm_text")
            confirm_ok = (
                confirm_raw is True
                and isinstance(confirm_text_raw, str)
                and confirm_text_raw.strip().upper() == "YOLO"
            )
            if not confirm_ok:
                return error_response(
                    status=400,
                    message=(
                        "Для включения YOLO требуется подтверждение "
                        "(yolo_confirm=true, yolo_confirm_text='YOLO')."
                    ),
                    error_type="invalid_request_error",
                    code="yolo_confirmation_required",
                )
        if yolo_armed:
            yolo_armed_at = api._utc_now_iso()
        else:
            yolo_armed_at = None
        api._save_policy_settings(
            profile=profile,
            yolo_armed=yolo_armed,
            yolo_armed_at=yolo_armed_at,
        )
        next_policy_profile = profile

    next_tools_state: dict[str, bool] | None = None
    if next_policy_profile is not None:
        profile_base = api._tools_state_for_profile(next_policy_profile, api._load_tools_state())
        api._save_tools_state(profile_base)
        next_tools_state = dict(profile_base)

    tools_raw = payload.get("tools")
    if tools_raw is not None:
        if not isinstance(tools_raw, dict):
            return error_response(
                status=400,
                message="tools должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        state_raw = tools_raw.get("state", tools_raw)
        if not isinstance(state_raw, dict):
            return error_response(
                status=400,
                message="tools.state должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        current_state = (
            next_tools_state if next_tools_state is not None else api._load_tools_state()
        )
        known_tool_keys = set(current_state.keys())
        next_state = dict(current_state)
        for key, raw_value in state_raw.items():
            if not isinstance(key, str) or key not in known_tool_keys:
                return error_response(
                    status=400,
                    message=f"Неизвестный tools ключ: {key}",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            if not isinstance(raw_value, bool):
                return error_response(
                    status=400,
                    message=f"tools.{key} должен быть bool.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            next_state[key] = raw_value
        api._save_tools_state(next_state)
        next_tools_state = dict(next_state)

    if next_tools_state is not None:
        agent_provider = cast(ScopedAgentProvider[api.AgentProtocol], request.app["agent_provider"])

        def _apply_tools(agent: api.AgentProtocol) -> None:
            update_tools_enabled = getattr(agent, "update_tools_enabled", None)
            if callable(update_tools_enabled):
                update_tools_enabled(next_tools_state)

        await agent_provider.apply_to_existing(_apply_tools)

    return json_response(api._build_settings_payload())


async def handle_ui_tts_speak(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )

    text_raw = payload.get("text")
    if not isinstance(text_raw, str) or not text_raw.strip():
        return error_response(
            status=400,
            message="text должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    voice_id_raw = payload.get("voice_id")
    if voice_id_raw is not None and not isinstance(voice_id_raw, str):
        return error_response(
            status=400,
            message="voice_id должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    voice_raw = payload.get("voice")
    if voice_raw is not None and not isinstance(voice_raw, str):
        return error_response(
            status=400,
            message="voice должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    format_raw = payload.get("format")
    if format_raw is not None and not isinstance(format_raw, str):
        return error_response(
            status=400,
            message="format должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    try:
        agent = await api._resolve_agent(request)
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=500,
            message=f"Не удалось получить agent для TTS: {exc}",
            error_type="internal_error",
            code="internal_error",
        )
    if agent is None:
        return error_response(
            status=409,
            message="Не выбрана модель. Выберите модель в UI и повторите.",
            error_type="configuration_error",
            code="model_not_selected",
        )

    args: dict[str, JSONValue] = {"text": text_raw.strip()}
    voice_value = (
        voice_raw.strip()
        if isinstance(voice_raw, str) and voice_raw.strip()
        else voice_id_raw.strip()
        if isinstance(voice_id_raw, str) and voice_id_raw.strip()
        else None
    )
    if voice_value is not None:
        args["voice"] = voice_value
        args["voice_id"] = voice_value
    if isinstance(format_raw, str) and format_raw.strip():
        args["format"] = format_raw.strip().lower()

    agent_lock = api._agent_lock_for_request(request)
    async with agent_lock:
        result = agent.call_tool("tts", args=args, raw_input="ui:tts")
    if not result.ok:
        return _tts_error_response(result.error or "TTS tool failed.")

    file_path_raw = result.data.get("file_path")
    format_value = str(result.data.get("format") or args.get("format") or "mp3").strip().lower()
    if format_value not in {"mp3", "wav"}:
        format_value = "mp3"
    if not isinstance(file_path_raw, str) or not file_path_raw.strip():
        return error_response(
            status=502,
            message="TTS tool не вернул путь к аудиофайлу.",
            error_type="upstream_error",
            code="tts_failed",
        )

    file_path = Path(file_path_raw).resolve()
    try:
        file_path.relative_to(_SANDBOX_AUDIO_ROOT)
    except ValueError:
        return error_response(
            status=502,
            message="TTS tool вернул недопустимый путь аудиофайла.",
            error_type="upstream_error",
            code="tts_failed",
        )
    if not file_path.exists() or not file_path.is_file():
        return error_response(
            status=502,
            message="Аудиофайл TTS не найден.",
            error_type="upstream_error",
            code="tts_failed",
        )

    try:
        audio_bytes = file_path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=502,
            message=f"Не удалось прочитать TTS-аудио: {exc}",
            error_type="upstream_error",
            code="tts_failed",
        )

    return web.Response(
        body=audio_bytes,
        content_type=f"audio/{format_value}",
        headers={
            "Content-Disposition": f'inline; filename="{file_path.name}"',
            "Cache-Control": "no-store",
        },
    )


async def handle_ui_stt_transcribe(request: web.Request) -> web.Response:
    api_key = api._resolve_provider_api_key("openai")
    if not api_key:
        return error_response(
            status=409,
            message="Не задан OpenAI API key для STT (env или Settings → API Keys).",
            error_type="configuration_error",
            code="stt_api_key_missing",
        )

    try:
        reader = await request.multipart()
    except Exception:  # noqa: BLE001
        return error_response(
            status=400,
            message="Ожидался multipart/form-data с полем audio.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    audio_bytes: bytes | None = None
    audio_filename = "recording.webm"
    audio_content_type = "application/octet-stream"
    language = "ru"
    while True:
        part = await reader.next()
        if part is None:
            break
        if not isinstance(part, BodyPartReader):
            continue
        name = str(getattr(part, "name", "") or "").strip()
        if not name:
            continue
        if name == "language":
            try:
                language_raw = await part.text()
            except Exception:  # noqa: BLE001
                language_raw = ""
            normalized_language = language_raw.strip()
            if normalized_language:
                language = normalized_language
            continue
        if name != "audio":
            continue
        audio_filename_raw = getattr(part, "filename", None)
        if isinstance(audio_filename_raw, str) and audio_filename_raw.strip():
            audio_filename = audio_filename_raw.strip()
        part_content_type = part.headers.get("Content-Type")
        if isinstance(part_content_type, str) and part_content_type.strip():
            audio_content_type = part_content_type.strip()
        chunks: list[bytes] = []
        total_size = 0
        while True:
            chunk = await part.read_chunk()
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > api.MAX_STT_AUDIO_BYTES:
                return error_response(
                    status=413,
                    message="Аудиофайл слишком большой.",
                    error_type="invalid_request_error",
                    code="payload_too_large",
                )
            chunks.append(chunk)
        if chunks:
            audio_bytes = b"".join(chunks)

    if audio_bytes is None:
        return error_response(
            status=400,
            message="Поле audio обязательно.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    try:
        requests_module = api._requests_module()
        response = requests_module.post(
            api.OPENAI_STT_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": "whisper-1",
                "language": language,
                "response_format": "json",
            },
            files={"file": (audio_filename, audio_bytes, audio_content_type)},
            timeout=api.MODEL_FETCH_TIMEOUT,
        )
    except Exception:  # noqa: BLE001
        return error_response(
            status=502,
            message="Не удалось связаться с STT-провайдером.",
            error_type="upstream_error",
            code="upstream_error",
        )

    if response.status_code >= 400:
        upstream_message = _openai_error_message(response)
        if response.status_code in {400, 415, 422}:
            return error_response(
                status=400,
                message=upstream_message or "Неподдерживаемый формат аудио.",
                error_type="invalid_request_error",
                code="unsupported_audio_format",
            )
        return error_response(
            status=502,
            message=upstream_message or "STT-провайдер вернул ошибку.",
            error_type="upstream_error",
            code="upstream_error",
        )

    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = None
    if not isinstance(payload, dict):
        return error_response(
            status=502,
            message="STT-провайдер вернул неожиданный ответ.",
            error_type="upstream_error",
            code="upstream_error",
        )
    text_raw = payload.get("text")
    if not isinstance(text_raw, str) or not text_raw.strip():
        return error_response(
            status=502,
            message="STT-провайдер не вернул текст распознавания.",
            error_type="upstream_error",
            code="upstream_error",
        )
    return json_response(
        {
            "text": text_raw.strip(),
            "model": "whisper-1",
            "language": language,
        }
    )
