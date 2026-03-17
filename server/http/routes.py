from __future__ import annotations

from aiohttp import web


def register_routes(app: web.Application) -> None:
    from server.http.handlers import (
        chat,
        decision,
        events,
        memory,
        models,
        plan,
        project,
        sessions,
        settings,
        slavik,
        ui_chat,
        workflow,
        workspace,
        workspaces_v2,
    )

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
    app.router.add_get("/ui", workspace.handle_ui_redirect)
    app.router.add_get("/ui/", workspace.handle_ui_index)
    app.router.add_get("/workspace", workspace.handle_workspace_index)
    app.router.add_get("/ui/workspace", workspace.handle_workspace_index)

    app.router.add_get("/ui/api/status", workflow.handle_ui_status)
    app.router.add_get("/ui/api/state", workflow.handle_ui_state)
    app.router.add_post("/ui/api/mode", workflow.handle_ui_mode)
    app.router.add_post("/ui/api/runtime/init", workflow.handle_ui_runtime_init)
    app.router.add_post("/ui/api/init", workflow.handle_ui_runtime_init)
    app.router.add_get("/ui/api/settings", settings.handle_ui_settings)
    app.router.add_post("/ui/api/settings", settings.handle_ui_settings_update)
    app.router.add_post("/ui/api/tts/speak", settings.handle_ui_tts_speak)
    app.router.add_post("/ui/api/stt/transcribe", settings.handle_ui_stt_transcribe)
    app.router.add_get("/ui/api/models", workspaces_v2.handle_ui_models)
    app.router.add_get("/ui/api/memory/conflicts", memory.handle_ui_memory_conflicts)
    app.router.add_post(
        "/ui/api/memory/conflicts/resolve",
        memory.handle_ui_memory_conflicts_resolve,
    )
    app.router.add_post("/ui/api/memory/triage/preview", memory.handle_ui_memory_triage_preview)
    app.router.add_post("/ui/api/memory/triage/apply", memory.handle_ui_memory_triage_apply)
    app.router.add_post("/ui/api/memory/triage/undo", memory.handle_ui_memory_triage_undo)
    app.router.add_post("/ui/api/tools/project", project.handle_ui_project_command)
    app.router.add_post("/ui/api/decision/respond", decision.handle_ui_decision_respond)
    app.router.add_post("/ui/api/plan/draft", plan.handle_ui_plan_draft)
    app.router.add_post("/ui/api/plan/approve", plan.handle_ui_plan_approve)
    app.router.add_post("/ui/api/plan/edit", plan.handle_ui_plan_edit)
    app.router.add_post("/ui/api/plan/execute", plan.handle_ui_plan_execute)
    app.router.add_post("/ui/api/plan/cancel", plan.handle_ui_plan_cancel)
    app.router.add_get("/ui/api/folders", sessions.handle_ui_folders_list)
    app.router.add_post("/ui/api/folders", sessions.handle_ui_folders_create)
    app.router.add_get("/ui/api/sessions", sessions.handle_ui_sessions_list)
    app.router.add_post("/ui/api/sessions", sessions.handle_ui_sessions_create)
    app.router.add_get("/ui/api/sessions/{session_id}", sessions.handle_ui_session_get)
    app.router.add_get(
        "/ui/api/sessions/{session_id}/history",
        sessions.handle_ui_session_history_get,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/output",
        sessions.handle_ui_session_output_get,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/files",
        sessions.handle_ui_session_files_get,
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
    app.router.add_get("/ui/api/settings/chats/export", sessions.handle_ui_chats_export)
    app.router.add_post("/ui/api/settings/chats/import", sessions.handle_ui_chats_import)

    app.router.add_get("/ui/api/workspaces", workspaces_v2.handle_ui_workspaces_list)
    app.router.add_post("/ui/api/workspaces", workspaces_v2.handle_ui_workspaces_create)
    app.router.add_get("/ui/api/workspaces/{workspace_id}", workspaces_v2.handle_ui_workspace_get)
    app.router.add_delete(
        "/ui/api/workspaces/{workspace_id}",
        workspaces_v2.handle_ui_workspace_delete,
    )
    app.router.add_get(
        "/ui/api/workspaces/{workspace_id}/root",
        workspaces_v2.handle_ui_workspace_root_get,
    )
    app.router.add_post(
        "/ui/api/workspaces/{workspace_id}/root/select",
        workspaces_v2.handle_ui_workspace_root_select,
    )
    app.router.add_get(
        "/ui/api/workspaces/{workspace_id}/activity",
        workspaces_v2.handle_ui_workspace_activity_get,
    )
    app.router.add_get(
        "/ui/api/workspaces/{workspace_id}/security",
        workspaces_v2.handle_ui_workspace_security_get,
    )
    app.router.add_post(
        "/ui/api/workspaces/{workspace_id}/security",
        workspaces_v2.handle_ui_workspace_security_post,
    )
    app.router.add_get(
        "/ui/api/workspaces/{workspace_id}/chats",
        workspaces_v2.handle_ui_workspace_chats_list,
    )
    app.router.add_post(
        "/ui/api/workspaces/{workspace_id}/chats",
        workspaces_v2.handle_ui_workspace_chats_create,
    )
    app.router.add_get(
        "/ui/api/workspaces/{workspace_id}/tree",
        workspaces_v2.handle_ui_workspace_tree,
    )
    app.router.add_get(
        "/ui/api/workspaces/{workspace_id}/git-diff",
        workspaces_v2.handle_ui_workspace_git_diff,
    )
    app.router.add_get(
        "/ui/api/workspaces/{workspace_id}/file",
        workspaces_v2.handle_ui_workspace_file_get,
    )
    app.router.add_put(
        "/ui/api/workspaces/{workspace_id}/file",
        workspaces_v2.handle_ui_workspace_file_put,
    )
    app.router.add_delete(
        "/ui/api/workspaces/{workspace_id}/file",
        workspaces_v2.handle_ui_workspace_file_delete,
    )
    app.router.add_post(
        "/ui/api/workspaces/{workspace_id}/file/create",
        workspaces_v2.handle_ui_workspace_file_create,
    )
    app.router.add_post(
        "/ui/api/workspaces/{workspace_id}/file/rename",
        workspaces_v2.handle_ui_workspace_file_rename,
    )
    app.router.add_post(
        "/ui/api/workspaces/{workspace_id}/file/move",
        workspaces_v2.handle_ui_workspace_file_move,
    )
    app.router.add_post(
        "/ui/api/workspaces/{workspace_id}/patch",
        workspaces_v2.handle_ui_workspace_patch,
    )
    app.router.add_post(
        "/ui/api/workspaces/{workspace_id}/run",
        workspaces_v2.handle_ui_workspace_run,
    )
    app.router.add_post(
        "/ui/api/workspaces/{workspace_id}/terminal/run",
        workspaces_v2.handle_ui_workspace_terminal_run,
    )
    app.router.add_post(
        "/ui/api/workspaces/{workspace_id}/index",
        workspaces_v2.handle_ui_workspace_index_run,
    )
    app.router.add_get("/ui/api/session/security", workspace.handle_ui_session_security_get)
    app.router.add_post("/ui/api/session/security", workspace.handle_ui_session_security_post)
    app.router.add_get("/ui/api/workspace/root", workspace.handle_ui_workspace_root_get)
    app.router.add_post("/ui/api/workspace/root/select", workspace.handle_ui_workspace_root_select)
    app.router.add_post("/ui/api/workspace/index", workspace.handle_ui_workspace_index_run)
    app.router.add_get("/ui/api/workspace/git-diff", workspace.handle_ui_workspace_git_diff)
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
        "/ui/api/workspace/terminal/run",
        workspace.handle_ui_workspace_terminal_run,
    )

    app.router.add_get("/ui/api/chats/{chat_id}", workspaces_v2.handle_ui_chat_get)
    app.router.add_delete("/ui/api/chats/{chat_id}", workspaces_v2.handle_ui_chat_delete)
    app.router.add_patch(
        "/ui/api/chats/{chat_id}/title",
        workspaces_v2.handle_ui_chat_title_patch,
    )
    app.router.add_get(
        "/ui/api/chats/{chat_id}/messages",
        workspaces_v2.handle_ui_chat_messages_get,
    )
    app.router.add_post("/ui/api/chats/{chat_id}/send", workspaces_v2.handle_ui_chat_send)
    app.router.add_post("/ui/api/chats/{chat_id}/model", workspaces_v2.handle_ui_chat_model_post)
    app.router.add_get("/ui/api/chats/{chat_id}/state", workspaces_v2.handle_ui_chat_state_get)
    app.router.add_post("/ui/api/chats/{chat_id}/mode", workspaces_v2.handle_ui_chat_mode_post)
    app.router.add_post(
        "/ui/api/chats/{chat_id}/runtime/init",
        workspaces_v2.handle_ui_chat_runtime_init,
    )
    app.router.add_post(
        "/ui/api/chats/{chat_id}/decision/respond",
        workspaces_v2.handle_ui_chat_decision_respond,
    )
    app.router.add_get(
        "/ui/api/chats/{chat_id}/events/stream",
        events.handle_ui_events_stream,
    )
    app.router.add_post(
        "/ui/api/chats/{chat_id}/plan/draft",
        workspaces_v2.handle_ui_chat_plan_draft,
    )
    app.router.add_post(
        "/ui/api/chats/{chat_id}/plan/approve",
        workspaces_v2.handle_ui_chat_plan_approve,
    )
    app.router.add_post(
        "/ui/api/chats/{chat_id}/plan/edit",
        workspaces_v2.handle_ui_chat_plan_edit,
    )
    app.router.add_post(
        "/ui/api/chats/{chat_id}/plan/execute",
        workspaces_v2.handle_ui_chat_plan_execute,
    )
    app.router.add_post(
        "/ui/api/chats/{chat_id}/plan/cancel",
        workspaces_v2.handle_ui_chat_plan_cancel,
    )
    app.router.add_post("/ui/api/chat/send", ui_chat.handle_ui_chat_send)
    app.router.add_get("/ui/api/events/stream", events.handle_ui_events_stream)
