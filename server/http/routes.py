from __future__ import annotations

from aiohttp import web


def register_routes(app: web.Application) -> None:
    from server import http_api as api

    app.router.add_get("/v1/models", api.handle_models)
    app.router.add_post("/v1/chat/completions", api.handle_chat_completions)
    app.router.add_get("/slavik/trace/{trace_id}", api.handle_trace)
    app.router.add_get("/slavik/tool-calls/{trace_id}", api.handle_tool_calls)
    app.router.add_post("/slavik/feedback", api.handle_feedback)
    app.router.add_post("/slavik/approve-session", api.handle_approve_session)
    app.router.add_post(
        "/slavik/admin/settings/security",
        api.handle_admin_security_settings_update,
    )
    app.router.add_get("/", api.handle_ui_index)
    app.router.add_get("/workspace", api.handle_workspace_index)
    app.router.add_get("/ui", api.handle_ui_redirect)
    app.router.add_get("/ui/", api.handle_ui_index)
    app.router.add_get("/ui/workspace", api.handle_workspace_index)
    app.router.add_get("/ui/api/status", api.handle_ui_status)
    app.router.add_get("/ui/api/state", api.handle_ui_state)
    app.router.add_post("/ui/api/mode", api.handle_ui_mode)
    app.router.add_post("/ui/api/plan/draft", api.handle_ui_plan_draft)
    app.router.add_post("/ui/api/plan/approve", api.handle_ui_plan_approve)
    app.router.add_post("/ui/api/plan/edit", api.handle_ui_plan_edit)
    app.router.add_post("/ui/api/plan/execute", api.handle_ui_plan_execute)
    app.router.add_post("/ui/api/plan/cancel", api.handle_ui_plan_cancel)
    app.router.add_get("/ui/api/settings", api.handle_ui_settings)
    app.router.add_post("/ui/api/settings", api.handle_ui_settings_update)
    app.router.add_get("/ui/api/memory/conflicts", api.handle_ui_memory_conflicts)
    app.router.add_post(
        "/ui/api/memory/conflicts/resolve",
        api.handle_ui_memory_conflicts_resolve,
    )
    app.router.add_post("/ui/api/stt/transcribe", api.handle_ui_stt_transcribe)
    app.router.add_get("/ui/api/settings/chats/export", api.handle_ui_chats_export)
    app.router.add_post("/ui/api/settings/chats/import", api.handle_ui_chats_import)
    app.router.add_get("/ui/api/models", api.handle_ui_models)
    app.router.add_get("/ui/api/folders", api.handle_ui_folders_list)
    app.router.add_post("/ui/api/folders", api.handle_ui_folders_create)
    app.router.add_get("/ui/api/sessions", api.handle_ui_sessions_list)
    app.router.add_post("/ui/api/sessions", api.handle_ui_sessions_create)
    app.router.add_get("/ui/api/sessions/{session_id}", api.handle_ui_session_get)
    app.router.add_get(
        "/ui/api/sessions/{session_id}/history",
        api.handle_ui_session_history_get,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/output",
        api.handle_ui_session_output_get,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/files",
        api.handle_ui_session_files_get,
    )
    app.router.add_post("/ui/api/decision/respond", api.handle_ui_decision_respond)
    app.router.add_get(
        "/ui/api/sessions/{session_id}/files/download",
        api.handle_ui_session_file_download,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/artifacts/download-all",
        api.handle_ui_session_artifacts_download_all,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/artifacts/{artifact_id}/download",
        api.handle_ui_session_artifact_download,
    )
    app.router.add_delete("/ui/api/sessions/{session_id}", api.handle_ui_session_delete)
    app.router.add_patch(
        "/ui/api/sessions/{session_id}/title",
        api.handle_ui_session_title_update,
    )
    app.router.add_put(
        "/ui/api/sessions/{session_id}/folder",
        api.handle_ui_session_folder_update,
    )
    app.router.add_post("/ui/api/session-model", api.handle_ui_session_model)
    app.router.add_get("/ui/api/session/security", api.handle_ui_session_security_get)
    app.router.add_post("/ui/api/session/security", api.handle_ui_session_security_post)
    app.router.add_get("/ui/api/workspace/root", api.handle_ui_workspace_root_get)
    app.router.add_post("/ui/api/workspace/root/select", api.handle_ui_workspace_root_select)
    app.router.add_post("/ui/api/workspace/index", api.handle_ui_workspace_index_run)
    app.router.add_get("/ui/api/workspace/git-diff", api.handle_ui_workspace_git_diff)
    app.router.add_get("/ui/api/workspace/tree", api.handle_ui_workspace_tree)
    app.router.add_get("/ui/api/workspace/file", api.handle_ui_workspace_file_get)
    app.router.add_put("/ui/api/workspace/file", api.handle_ui_workspace_file_put)
    app.router.add_post("/ui/api/workspace/run", api.handle_ui_workspace_run)
    app.router.add_post("/ui/api/chat/send", api.handle_ui_chat_send)
    app.router.add_post("/ui/api/tools/project", api.handle_ui_project_command)
    app.router.add_get("/ui/api/events/stream", api.handle_ui_events_stream)
