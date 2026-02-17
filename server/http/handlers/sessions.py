from __future__ import annotations

import io
import zipfile
from typing import Literal, cast

from aiohttp import web

from server import http_api as api
from server.http.common.responses import error_response, json_response
from server.http_api import (
    SUPPORTED_MODEL_PROVIDERS,
    UI_SESSION_HEADER,
    _artifact_file_payload,
    _artifact_mime_from_ext,
    _closest_model_suggestion,
    _ensure_session_owned,
    _normalize_provider,
    _normalize_ui_decision,
    _parse_imported_session,
    _request_principal_id,
    _resolve_ui_session_id_for_principal,
    _resolve_workspace_file,
    _safe_zip_entry_name,
    _sanitize_download_filename,
    _serialize_persisted_session,
    _session_forbidden_response,
    _utc_iso,
)
from server.ui_hub import UIHub
from server.ui_session_storage import PersistedSession
from shared.models import JSONValue


async def handle_ui_chats_export(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    principal_id = _request_principal_id(request)
    if principal_id is None:
        return error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
    sessions = await hub.export_sessions(principal_id)
    payload_sessions = [_serialize_persisted_session(item) for item in sessions]
    return json_response(
        {
            "exported_at": _utc_iso(),
            "count": len(payload_sessions),
            "sessions": payload_sessions,
        },
    )


async def handle_ui_chats_import(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    principal_id = _request_principal_id(request)
    if principal_id is None:
        return error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
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
    sessions_raw = payload.get("sessions")
    if not isinstance(sessions_raw, list):
        return error_response(
            status=400,
            message="sessions должен быть списком.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    mode: Literal["replace", "merge"] = "replace"
    raw_mode = payload.get("mode")
    if isinstance(raw_mode, str):
        normalized_mode = raw_mode.strip().lower()
        if normalized_mode in {"replace", "merge"}:
            mode = cast(Literal["replace", "merge"], normalized_mode)
        else:
            return error_response(
                status=400,
                message="mode должен быть replace или merge.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )

    sessions: list[PersistedSession] = []
    for index, item in enumerate(sessions_raw):
        parsed = _parse_imported_session(item, principal_id=principal_id)
        if parsed is None:
            return error_response(
                status=400,
                message=f"sessions[{index}] имеет некорректный формат.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        sessions.append(parsed)
    imported = await hub.import_sessions(sessions, principal_id=principal_id, mode=mode)
    return json_response(
        {
            "imported": imported,
            "mode": mode,
        },
    )


async def handle_ui_models(request: web.Request) -> web.Response:
    provider_query = request.query.get("provider", "").strip().lower()
    providers: list[str]
    if provider_query:
        normalized = _normalize_provider(provider_query)
        if normalized is None:
            return error_response(
                status=400,
                message=f"Неизвестный провайдер: {provider_query}",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        providers = [normalized]
    else:
        providers = sorted(SUPPORTED_MODEL_PROVIDERS)
    payload_items: list[dict[str, JSONValue]] = []
    for provider in providers:
        models, error_text = api._fetch_provider_models(provider)
        payload_items.append({"provider": provider, "models": models, "error": error_text})
    return json_response({"providers": payload_items})


async def handle_ui_session_model(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
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
    provider_raw = str(payload.get("provider") or "").strip()
    model_raw = str(payload.get("model") or "").strip()
    if not provider_raw or not model_raw:
        return error_response(
            status=400,
            message="Нужны provider и model.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    provider = _normalize_provider(provider_raw)
    if provider is None:
        return error_response(
            status=400,
            message=f"Неизвестный провайдер: {provider_raw}",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    models, fetch_error = api._fetch_provider_models(provider)
    if fetch_error:
        return error_response(
            status=502,
            message=fetch_error,
            error_type="provider_error",
            code="provider_models_unavailable",
        )
    if model_raw not in models:
        suggestion = _closest_model_suggestion(model_raw, models)
        details: dict[str, JSONValue] = {
            "provider": provider,
            "model": model_raw,
            "suggestion": suggestion,
            "available_count": len(models),
        }
        message = (
            f"сам придумал, сам и страдай. "
            f"Модель '{model_raw}' не найдена у провайдера '{provider}'."
        )
        if suggestion:
            message = f"{message} Возможно, вы имели в виду '{suggestion}'."
        return error_response(
            status=404,
            message=message,
            error_type="invalid_request_error",
            code="model_not_found",
            details=details,
        )
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None or session_id is None:
        return session_error or _session_forbidden_response()
    await hub.set_session_model(session_id, provider, model_raw)
    response = json_response(
        {"session_id": session_id, "selected_model": {"provider": provider, "model": model_raw}}
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_sessions_list(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    principal_id = _request_principal_id(request)
    if principal_id is None:
        return error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
    sessions = await hub.list_sessions(principal_id)
    serialized_sessions: list[dict[str, JSONValue]] = [
        {
            "session_id": item["session_id"],
            "title": item["title"],
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
            "message_count": item["message_count"],
            "title_override": item["title_override"],
            "folder_id": item["folder_id"],
        }
        for item in sessions
    ]
    return json_response({"sessions": serialized_sessions})


async def handle_ui_folders_list(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    folders = await hub.list_folders()
    return json_response({"folders": folders})


async def handle_ui_folders_create(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
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
    name_raw = payload.get("name")
    if not isinstance(name_raw, str) or not name_raw.strip():
        return error_response(
            status=400,
            message="name должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        folder = await hub.create_folder(name_raw)
    except ValueError:
        return error_response(
            status=400,
            message="name должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    return json_response({"folder": folder})


async def handle_ui_sessions_create(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    principal_id = _request_principal_id(request)
    if principal_id is None:
        return error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
    session_id = await hub.create_session(principal_id)
    session = await hub.get_session(session_id)
    if session is None:
        return error_response(
            status=500,
            message="Failed to create session.",
            error_type="internal_error",
            code="session_create_failed",
        )
    response = json_response({"session": session})
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error
    session = await hub.get_session(session_id)
    if session is None:
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    raw_decision = session.get("decision")
    session["decision"] = _normalize_ui_decision(raw_decision, session_id=session_id)
    response = json_response({"session": session})
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_history_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error
    messages = await hub.get_session_history(session_id)
    if messages is None:
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return json_response({"session_id": session_id, "messages": messages})


async def handle_ui_session_output_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error
    output = await hub.get_session_output(session_id)
    if output is None:
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return json_response({"session_id": session_id, "output": output})


async def handle_ui_session_files_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error
    files = await hub.get_session_files(session_id)
    if files is None:
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return json_response({"session_id": session_id, "files": files})


async def handle_ui_session_file_download(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error
    files = await hub.get_session_files(session_id)
    if files is None:
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    path_raw = request.query.get("path", "").strip()
    if not path_raw:
        return error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if path_raw not in files:
        return error_response(
            status=404,
            message="File not found in session.",
            error_type="invalid_request_error",
            code="session_file_not_found",
        )
    try:
        file_path = _resolve_workspace_file(path_raw)
        content = file_path.read_bytes()
    except FileNotFoundError:
        return error_response(
            status=404,
            message="File not found.",
            error_type="invalid_request_error",
            code="session_file_not_found",
        )
    except ValueError as exc:
        return error_response(
            status=400,
            message=f"Invalid path: {exc}",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    safe_name = _sanitize_download_filename(path_raw)
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    mime = _artifact_mime_from_ext(ext)
    response = web.Response(body=content, content_type=mime)
    response.headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_artifact_download(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    artifact_id = request.match_info.get("artifact_id", "").strip()
    if not session_id or not artifact_id:
        return error_response(
            status=400,
            message="session_id и artifact_id обязательны.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error
    artifacts = await hub.get_session_artifacts(session_id)
    if artifacts is None:
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    target = next(
        (
            artifact
            for artifact in artifacts
            if isinstance(artifact.get("id"), str) and str(artifact.get("id")) == artifact_id
        ),
        None,
    )
    if target is None:
        return error_response(
            status=404,
            message="Artifact not found.",
            error_type="invalid_request_error",
            code="artifact_not_found",
        )
    try:
        file_name, file_content, mime = _artifact_file_payload(target)
    except ValueError as exc:
        return error_response(
            status=400,
            message=f"Artifact is not downloadable file: {exc}",
            error_type="invalid_request_error",
            code="artifact_not_file",
        )

    response = web.Response(body=file_content.encode("utf-8"), content_type=mime)
    response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_artifacts_download_all(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error
    artifacts = await hub.get_session_artifacts(session_id)
    if artifacts is None:
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )

    file_items: list[tuple[str, str, str]] = []
    for artifact in artifacts:
        try:
            file_items.append(_artifact_file_payload(artifact))
        except ValueError:
            continue
    if not file_items:
        return error_response(
            status=404,
            message="No file artifacts for download.",
            error_type="invalid_request_error",
            code="artifact_not_file",
        )

    force_zip = request.query.get("format", "").strip().lower() == "zip"
    if len(file_items) == 1 and not force_zip:
        file_name, file_content, mime = file_items[0]
        response = web.Response(body=file_content.encode("utf-8"), content_type=mime)
        response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
        response.headers[UI_SESSION_HEADER] = session_id
        return response

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        used_names: set[str] = set()
        for file_name, file_content, _ in file_items:
            entry_name = _safe_zip_entry_name(file_name)
            base_name = entry_name
            suffix = 1
            while entry_name in used_names:
                stem, dot, ext = base_name.partition(".")
                if dot:
                    entry_name = f"{stem}_{suffix}.{ext}"
                else:
                    entry_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(entry_name)
            archive.writestr(entry_name, file_content)
    zip_bytes = zip_buffer.getvalue()
    response = web.Response(body=zip_bytes, content_type="application/zip")
    response.headers["Content-Disposition"] = 'attachment; filename="artifacts.zip"'
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_title_update(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error
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
    title_raw = payload.get("title")
    if not isinstance(title_raw, str) or not title_raw.strip():
        return error_response(
            status=400,
            message="title должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        result = await hub.set_session_title(session_id, title_raw)
    except KeyError:
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    except ValueError:
        return error_response(
            status=400,
            message="title должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    return json_response(result)


async def handle_ui_session_folder_update(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error
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
    folder_raw = payload.get("folder_id")
    folder_id: str | None
    if folder_raw is None:
        folder_id = None
    elif isinstance(folder_raw, str):
        folder_id = folder_raw.strip() or None
    else:
        return error_response(
            status=400,
            message="folder_id должен быть строкой или null.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        result = await hub.assign_session_folder(session_id, folder_id)
    except KeyError as exc:
        if "folder" in str(exc).lower():
            return error_response(
                status=404,
                message="Folder not found.",
                error_type="invalid_request_error",
                code="folder_not_found",
            )
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return json_response(result)


async def handle_ui_session_delete(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error
    deleted = await hub.delete_session(session_id)
    if not deleted:
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return json_response({"session_id": session_id, "deleted": True})
