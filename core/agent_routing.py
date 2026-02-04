from __future__ import annotations

# ruff: noqa: F401
# mypy: ignore-errors
from core.approval_policy import ApprovalRequired
from core.decision.handler import DecisionContext, DecisionRequired
from core.mwv.models import StopReasonCode
from core.mwv.routing import classify_request
from core.skills.index import SkillMatchDecision
from shared.models import LLMMessage


class AgentRoutingMixin:
    _last_user_input: str | None
    last_reasoning: str | None

    def respond(self, messages: list[LLMMessage]) -> str:
        if not messages:
            return "[Пустое сообщение]"

        last_content = messages[-1].content.strip()
        self._last_user_input = last_content
        record_in_history = self._should_record_in_history(last_content)
        try:
            if record_in_history:
                self._append_short_term(messages)
            self.tracer.log("user_input", last_content)
            self._reset_approval_state()
            self.last_reasoning = None
            self._reset_workspace_diffs()

            if last_content.lower().startswith("авто") or last_content.startswith("/auto"):
                auto_goal = last_content.replace("/auto", "").strip()
                result = self.handle_auto_command(auto_goal)
                self._log_chat_interaction(raw_input=last_content, response_text=result)
                if record_in_history:
                    self._append_short_term([LLMMessage(role="assistant", content=result)])
                return result

            if last_content.startswith("/"):
                return self.handle_tool_command(last_content)

            decision = classify_request(
                messages,
                last_content,
                context={"safe_mode": bool(self.tools_enabled.get("safe_mode", False))},
                skill_index=self.skill_index,
            )
            self._apply_skill_decision(decision.skill_decision)
            self.tracer.log(
                "routing_decision",
                decision.route,
                {"reason": decision.reason, "flags": decision.risk_flags},
            )
            if decision.skill_decision and decision.skill_decision.status == "deprecated":
                response = self._format_skill_block(decision.skill_decision)
                if self.memory_config.auto_save_dialogue:
                    self.save_to_memory(last_content, response)
                self._log_chat_interaction(raw_input=last_content, response_text=response)
                if record_in_history:
                    self._append_short_term([LLMMessage(role="assistant", content=response)])
                return response
            decision_packet = self.decision_handler.evaluate(
                DecisionContext(
                    user_input=last_content,
                    route=decision.route,
                    reason=decision.reason,
                    risk_flags=list(decision.risk_flags),
                    skill_decision=decision.skill_decision,
                ),
            )
            if decision_packet is not None:
                return self._handle_decision_packet(
                    decision_packet,
                    raw_input=last_content,
                    record_in_history=record_in_history,
                )
            if decision.route == "mwv":
                if decision.skill_decision and decision.skill_decision.status == "no_match":
                    self._record_unknown_inbox(last_content, decision)
                    self._record_unknown_skill_candidate(last_content, decision)
                return self._run_mwv_flow(messages, last_content, decision, record_in_history)
            return self._run_chat_response(messages, last_content, record_in_history)
        except ApprovalRequired as exc:
            return self._handle_approval_required(
                exc.request,
                raw_input=last_content,
                record_in_history=record_in_history,
            )
        except DecisionRequired as exc:
            return self._handle_decision_packet(
                exc.packet,
                raw_input=last_content,
                record_in_history=record_in_history,
            )
        except Exception as exc:
            self.logger.exception("Agent.respond error: %s", exc)
            self.tracer.log("error", f"Ошибка Agent.respond: {exc}")
            error_text = f"[Ошибка ответа: {exc}]"
            try:
                self._log_chat_interaction(raw_input=last_content, response_text=error_text)
            except Exception as log_exc:  # noqa: BLE001
                self.logger.error("Ошибка записи InteractionLog: %s", log_exc)
            if record_in_history:
                self._append_short_term([LLMMessage(role="assistant", content=error_text)])
            return error_text

    def _run_chat_response(
        self,
        messages: list[LLMMessage],
        last_content: str,
        record_in_history: bool,
    ) -> str:
        try:
            self.tracer.log("reasoning_start", "Генерация ответа моделью")
            policy_application = self._apply_policies(last_content)
            messages_with_context = self._build_context_messages(self.short_term, last_content)
            messages_with_context = self._append_policy_instructions(
                messages_with_context,
                policy_application,
            )
            reply = self._get_main_brain().generate(messages_with_context)
            reviewed = self._review_answer(reply.text)
            if self.main_config and self.main_config.thinking_enabled:
                self.last_reasoning = reply.reasoning
            self.tracer.log("reasoning_end", "Ответ получен", {"reply_preview": reviewed[:120]})
            response_text = self._append_report_block(
                reviewed,
                route="chat",
                trace_id=None,
                attempts=None,
                verifier=None,
                next_steps=None,
                stop_reason_code=None,
            )
            if self.memory_config.auto_save_dialogue:
                self.save_to_memory(last_content, response_text)
            self._log_chat_interaction(
                raw_input=last_content,
                response_text=response_text,
                applied_policy_ids=policy_application.applied_policy_ids,
            )
            if record_in_history:
                self._append_short_term([LLMMessage(role="assistant", content=response_text)])
            return response_text
        except Exception as exc:  # noqa: BLE001
            self.logger.error("LLM error: %s", exc)
            self.tracer.log("error", f"Ошибка модели: {exc}")
            error_text = f"[Ошибка модели: {exc}]"
            self._log_chat_interaction(raw_input=last_content, response_text=error_text)
            if record_in_history:
                self._append_short_term([LLMMessage(role="assistant", content=error_text)])
            return error_text

    def _apply_skill_decision(self, decision: SkillMatchDecision | None) -> None:
        self._last_skill_match = None
        if decision is None:
            self.tracer.log("skill_match", "none")
            return
        if decision.status == "matched" and decision.match is not None:
            self._last_skill_match = decision.match
            self._inc_metric("skill_match_hit")
            self.tracer.log(
                "skill_match",
                decision.match.entry.id,
                {"pattern": decision.match.pattern},
            )
            return
        if decision.status == "deprecated" and decision.match is not None:
            self._inc_metric("deprecated_count")
            self.tracer.log(
                "skill_match",
                "deprecated",
                {
                    "skill_id": decision.match.entry.id,
                    "replaced_by": decision.replaced_by or "",
                },
            )
            return
        if decision.status == "ambiguous":
            self._inc_metric("ambiguous_count")
            self.tracer.log(
                "skill_match",
                "ambiguous",
                {"candidates": [match.entry.id for match in decision.alternatives]},
            )
            return
        if decision.status == "no_match":
            self._inc_metric("skill_match_miss")
        self.tracer.log("skill_match", "none")

    def _format_skill_block(self, decision: SkillMatchDecision) -> str:
        if decision.status == "deprecated" and decision.match is not None:
            replaced = decision.replaced_by or "нет замены"
            return self._format_stop_response(
                what="Навык deprecated и заблокирован",
                why=f"skill_id={decision.match.entry.id}; replaced_by={replaced}",
                next_steps=[
                    "Укажи новый skill_id или замену.",
                    "Переформулируй запрос.",
                ],
                stop_reason_code=StopReasonCode.BLOCKED_SKILL_DEPRECATED,
                route="blocked",
            )
        if decision.status == "ambiguous":
            ids = [match.entry.id for match in decision.alternatives]
            listed = ", ".join(ids) if ids else "unknown"
            return self._format_stop_response(
                what="Найдено несколько подходящих навыков",
                why=f"candidates={listed}",
                next_steps=[
                    "Укажи нужный skill_id.",
                    "Уточни запрос, чтобы матч был однозначным.",
                ],
                stop_reason_code=StopReasonCode.BLOCKED_SKILL_AMBIGUOUS,
                route="blocked",
            )
        return "Навык не может быть применен."
