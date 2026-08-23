from __future__ import annotations

from aiohttp import web


def register_routes(app: web.Application) -> None:
    from server.http.handlers import (
        auth,
        chat,
        decision,
        desktop,
        embeddings,
        events,
        health,
        memory,
        models,
        plan,
        project,
        sessions,
        settings,
        slavik,
        terminal,
        ui_chat,
        workflow,
        workspace,
    )

    app.router.add_get("/healthz", health.handle_health)

    app.router.add_get("/ui/api/auth/status", auth.handle_ui_auth_status)
    app.router.add_post("/ui/api/auth/login", auth.handle_ui_auth_login)
    app.router.add_post("/ui/api/auth/logout", auth.handle_ui_auth_logout)

    app.router.add_get("/v1/models", models.handle_models)
    app.router.add_post("/v1/chat/completions", chat.handle_chat_completions)
    app.router.add_get("/slavik/trace/{trace_id}", slavik.handle_trace)
    app.router.add_get("/slavik/tool-calls/{trace_id}", slavik.handle_tool_calls)
    app.router.add_post("/slavik/feedback", slavik.handle_feedback)
    app.router.add_post("/slavik/approve-session", slavik.handle_approve_session)
    app.router.add_post(
        "/slavik/admin/settings/security",
        settings.handle_admin_security_settings_update,
    )
    app.router.add_get("/", workspace.handle_ui_index)
    app.router.add_get("/workspace", workspace.handle_workspace_index)
    app.router.add_get("/ui", workspace.handle_ui_redirect)
    app.router.add_get("/ui/", workspace.handle_ui_index)
    app.router.add_get("/ui/workspace", workspace.handle_workspace_index)
    app.router.add_get("/ui/api/status", workflow.handle_ui_status)
    app.router.add_get("/ui/api/trace/{trace_id}", slavik.handle_ui_trace)
    app.router.add_get("/ui/api/tool-calls/{trace_id}", slavik.handle_ui_tool_calls)
    app.router.add_post("/ui/api/feedback", slavik.handle_ui_feedback)
    app.router.add_get("/ui/api/state", workflow.handle_ui_state)
    app.router.add_post("/ui/api/mode", workflow.handle_ui_mode)
    app.router.add_post("/ui/api/runtime/init", workflow.handle_ui_runtime_init)
    app.router.add_post("/ui/api/init", workflow.handle_ui_runtime_init)
    app.router.add_post("/ui/api/plan/draft", plan.handle_ui_plan_draft)
    app.router.add_post("/ui/api/plan/approve", plan.handle_ui_plan_approve)
    app.router.add_post("/ui/api/plan/edit", plan.handle_ui_plan_edit)
    app.router.add_post("/ui/api/plan/execute", plan.handle_ui_plan_execute)
    app.router.add_post("/ui/api/plan/cancel", plan.handle_ui_plan_cancel)
    app.router.add_get("/ui/api/settings", settings.handle_ui_settings)
    app.router.add_post("/ui/api/settings", settings.handle_ui_settings_update)
    app.router.add_get("/ui/api/embeddings/status", embeddings.handle_embeddings_status)
    app.router.add_post("/ui/api/embeddings/download", embeddings.handle_embeddings_download)
    app.router.add_get("/ui/api/memory/conflicts", memory.handle_ui_memory_conflicts)
    app.router.add_get("/ui/api/memory/pinned", memory.handle_ui_memory_pinned)
    app.router.add_post("/ui/api/memory/pin", memory.handle_ui_memory_pin)
    app.router.add_post("/ui/api/memory/unpin", memory.handle_ui_memory_unpin)
    app.router.add_post(
        "/ui/api/memory/conflicts/resolve",
        memory.handle_ui_memory_conflicts_resolve,
    )
    app.router.add_post("/ui/api/memory/triage/preview", memory.handle_ui_memory_triage_preview)
    app.router.add_post("/ui/api/memory/triage/apply", memory.handle_ui_memory_triage_apply)
    app.router.add_post("/ui/api/memory/triage/undo", memory.handle_ui_memory_triage_undo)
    app.router.add_post("/ui/api/tts/speak", settings.handle_ui_tts_speak)
    app.router.add_post("/ui/api/stt/transcribe", settings.handle_ui_stt_transcribe)
    app.router.add_get("/ui/api/settings/chats/export", sessions.handle_ui_chats_export)
    app.router.add_post("/ui/api/settings/chats/import", sessions.handle_ui_chats_import)
    app.router.add_get("/ui/api/models", sessions.handle_ui_models)
    app.router.add_post("/ui/api/local/ollama/start", sessions.handle_ui_local_ollama_start)
    app.router.add_get("/ui/api/folders", sessions.handle_ui_folders_list)
    app.router.add_post("/ui/api/folders", sessions.handle_ui_folders_create)
    app.router.add_get("/ui/api/sessions", sessions.handle_ui_sessions_list)
    app.router.add_post("/ui/api/sessions", sessions.handle_ui_sessions_create)
    app.router.add_get("/ui/api/sessions/{session_id}", sessions.handle_ui_session_get)
    app.router.add_get(
        "/ui/api/sessions/{session_id}/history",
        sessions.handle_ui_session_history_get,
    )
    app.router.add_delete(
        "/ui/api/sessions/{session_id}/messages/last",
        sessions.handle_ui_session_messages_last_delete,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/output",
        sessions.handle_ui_session_output_get,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/files",
        sessions.handle_ui_session_files_get,
    )
    app.router.add_post("/ui/api/decision/respond", decision.handle_ui_decision_respond)
    app.router.add_get(
        "/ui/api/desktop/approvals",
        desktop.handle_desktop_approval_rules_list,
    )
    app.router.add_post(
        "/ui/api/desktop/approvals",
        desktop.handle_desktop_approval_rule_create,
    )
    app.router.add_post(
        "/ui/api/desktop/approvals/reset-invalid",
        desktop.handle_desktop_approval_rules_reset_invalid,
    )
    app.router.add_patch(
        "/ui/api/desktop/approvals/{rule_id}",
        desktop.handle_desktop_approval_rule_update,
    )
    app.router.add_delete(
        "/ui/api/desktop/approvals/{rule_id}",
        desktop.handle_desktop_approval_rule_delete,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/files/download",
        sessions.handle_ui_session_file_download,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/artifacts/download-all",
        sessions.handle_ui_session_artifacts_download_all,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/artifacts/{artifact_id}/download",
        sessions.handle_ui_session_artifact_download,
    )
    app.router.add_delete("/ui/api/sessions/{session_id}", sessions.handle_ui_session_delete)
    app.router.add_patch(
        "/ui/api/sessions/{session_id}/title",
        sessions.handle_ui_session_title_update,
    )
    app.router.add_put(
        "/ui/api/sessions/{session_id}/folder",
        sessions.handle_ui_session_folder_update,
    )
    app.router.add_post("/ui/api/session-model", sessions.handle_ui_session_model)
    app.router.add_get("/ui/api/session/security", workspace.handle_ui_session_security_get)
    app.router.add_post("/ui/api/session/security", workspace.handle_ui_session_security_post)
    app.router.add_get("/ui/api/workspace/root", workspace.handle_ui_workspace_root_get)
    app.router.add_post("/ui/api/workspace/root/select", workspace.handle_ui_workspace_root_select)
    app.router.add_post("/ui/api/workspace/index", workspace.handle_ui_workspace_index_run)
    app.router.add_get("/ui/api/workspace/git-diff", workspace.handle_ui_workspace_git_diff)
    app.router.add_get("/ui/api/workspace/browse", workspace.handle_ui_workspace_browse)
    app.router.add_get("/ui/api/workspace/git/status", workspace.handle_ui_workspace_git_status)
    app.router.add_post("/ui/api/workspace/git/stage", workspace.handle_ui_workspace_git_stage)
    app.router.add_post("/ui/api/workspace/git/unstage", workspace.handle_ui_workspace_git_unstage)
    app.router.add_post("/ui/api/workspace/git/commit", workspace.handle_ui_workspace_git_commit)
    app.router.add_get("/ui/api/workspace/tree", workspace.handle_ui_workspace_tree)
    app.router.add_get("/ui/api/workspace/file", workspace.handle_ui_workspace_file_get)
    app.router.add_put("/ui/api/workspace/file", workspace.handle_ui_workspace_file_put)
    app.router.add_post("/ui/api/workspace/file/create", workspace.handle_ui_workspace_file_create)
    app.router.add_post("/ui/api/workspace/file/rename", workspace.handle_ui_workspace_file_rename)
    app.router.add_post("/ui/api/workspace/file/move", workspace.handle_ui_workspace_file_move)
    app.router.add_delete("/ui/api/workspace/file", workspace.handle_ui_workspace_file_delete)
    app.router.add_post("/ui/api/workspace/patch", workspace.handle_ui_workspace_patch)
    app.router.add_post("/ui/api/workspace/run", workspace.handle_ui_workspace_run)
    app.router.add_post(
        "/ui/api/workspace/terminal/run", workspace.handle_ui_workspace_terminal_run
    )
    app.router.add_post("/ui/api/terminal", terminal.handle_ui_terminal_create)
    app.router.add_get("/ui/api/terminal", terminal.handle_ui_terminal_get)
    app.router.add_get("/ui/api/terminal/stream", terminal.handle_ui_terminal_stream)
    app.router.add_post("/ui/api/terminal/input", terminal.handle_ui_terminal_input)
    app.router.add_post("/ui/api/terminal/resize", terminal.handle_ui_terminal_resize)
    app.router.add_post("/ui/api/terminal/close", terminal.handle_ui_terminal_close)
    app.router.add_post("/ui/api/chat/send", ui_chat.handle_ui_chat_send)
    app.router.add_post("/ui/api/chat/cancel", ui_chat.handle_ui_chat_cancel)
    app.router.add_post("/ui/api/tools/project", project.handle_ui_project_command)
    app.router.add_get("/ui/api/chat/events/{session_id}", events.handle_ui_chat_events_stream)
