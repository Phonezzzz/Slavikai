from __future__ import annotations

# ruff: noqa: F401
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Literal

from core.approval_policy import ApprovalRequired
from core.decision.handler import DecisionContext, DecisionRequired
from core.mwv.models import StopReasonCode
from core.mwv.routing import RouteDecision, classify_request
from core.skills.index import SkillMatchDecision
from core.tool_loop import AgentToolLoop
from llm.types import LLMResult, LLMStreamChunk, ToolSpec, WebSearchEvidence
from shared.models import LLMMessage, ToolRequest, ToolResult

if TYPE_CHECKING:
    import logging

    from config.memory_config import MemoryConfig
    from core.approval_policy import ApprovalRequest
    from core.decision.handler import DecisionHandler
    from core.decision.models import DecisionPacket
    from core.mwv.models import VerificationResult
    from core.rule_engine import PolicyApplication
    from core.skills.index import SkillIndex, SkillMatch
    from core.tool_gateway import ToolGateway
    from core.tracer import Tracer
    from llm.brain_base import Brain
    from llm.types import ModelConfig
    from shared.models import JSONValue
    from tools.tool_registry import ToolRegistry


class AgentRoutingMixin:
    if TYPE_CHECKING:
        tracer: Tracer
        logger: logging.Logger
        tools_enabled: dict[str, bool]
        tool_registry: ToolRegistry
        skill_index: SkillIndex | None
        decision_handler: DecisionHandler
        memory_config: MemoryConfig
        short_term: list[LLMMessage]
        main_config: ModelConfig | None
        _last_skill_match: SkillMatch | None

        def _should_record_in_history(self, user_input: str) -> bool: ...
        def _append_short_term(
            self,
            messages: list[LLMMessage],
            *,
            history: list[LLMMessage] | None = None,
        ) -> None: ...
        def _reset_approval_state(self) -> None: ...
        def _reset_workspace_diffs(self) -> None: ...
        def handle_tool_command(self, command: str) -> str: ...
        def handle_auto_command(self, goal: str, *, command_lane: bool = False) -> str: ...
        def save_to_memory(self, user_text: str, assistant_text: str) -> None: ...
        def is_explicit_memory_request(self, text: str) -> bool: ...
        def remember_explicit_text(
            self,
            text: str,
            *,
            source_kind: str,
            source_id: str | None = None,
            lang_hint: str | None = None,
        ) -> str: ...
        def _log_chat_interaction(
            self,
            raw_input: str,
            response_text: str,
            *,
            retrieved_memory_ids: list[str] | None = None,
            applied_policy_ids: list[str] | None = None,
        ) -> str: ...
        def _run_mwv_flow(
            self,
            messages: list[LLMMessage],
            last_content: str,
            route_decision: RouteDecision,
            record_in_history: bool,
        ) -> str: ...
        def _handle_approval_required(
            self,
            request: ApprovalRequest,
            *,
            raw_input: str,
            record_in_history: bool = False,
            command_lane: bool = False,
            source_endpoint: str | None = None,
            resume_payload: dict[str, JSONValue] | None = None,
        ) -> str: ...
        def _handle_decision_packet(
            self,
            packet: DecisionPacket,
            *,
            raw_input: str,
            record_in_history: bool,
        ) -> str: ...
        def _record_unknown_inbox(self, user_input: str, decision: RouteDecision) -> None: ...
        def _record_unknown_skill_candidate(
            self,
            user_input: str,
            decision: RouteDecision,
        ) -> None: ...
        def _apply_policies(self, user_input: str) -> PolicyApplication: ...
        def _build_context_messages(
            self,
            short_term: list[LLMMessage],
            user_input: str,
        ) -> list[LLMMessage]: ...
        def _append_policy_instructions(
            self,
            messages: list[LLMMessage],
            policy_application: PolicyApplication,
        ) -> list[LLMMessage]: ...
        def _get_main_brain(self) -> Brain: ...
        def _review_answer(self, raw_answer: str) -> str: ...
        def _call_tool_logged(
            self,
            raw_input: str,
            request: ToolRequest,
            *,
            safe_mode_override: bool | None = None,
        ) -> ToolResult: ...
        def _build_tool_gateway(
            self,
            *,
            pre_call: Callable[[ToolRequest], object | None] | None = None,
            post_call: Callable[[ToolRequest, ToolResult, object | None], None] | None = None,
            safe_mode_override: bool | None = None,
        ) -> ToolGateway: ...
        def _append_report_block(
            self,
            content: str,
            *,
            route: str,
            trace_id: str | None,
            attempts: tuple[int, int] | None,
            verifier: VerificationResult | None,
            next_steps: list[str] | None,
            stop_reason_code: StopReasonCode | None,
            plan_summary: str | None = None,
            execution_summary: str | None = None,
        ) -> str: ...
        def _inc_metric(self, metric_key: str) -> None: ...
        def _format_stop_response(
            self,
            *,
            what: str,
            why: str,
            next_steps: list[str],
            stop_reason_code: StopReasonCode,
            route: str,
            trace_id: str | None = None,
            attempts: tuple[int, int] | None = None,
            verifier: VerificationResult | None = None,
            plan_summary: str | None = None,
            execution_summary: str | None = None,
        ) -> str: ...

    _last_user_input: str | None
    last_reasoning: str | None
    last_stream_response_raw: str | None
    _WEB_CLAIM_MARKERS = (
        "проверил в интернете",
        "нашёл в сети",
        "нашел в сети",
        "according to web",
        "search found",
        "i checked",
    )

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

            if last_content.startswith("/"):
                return self.handle_tool_command(last_content)

            if self.is_explicit_memory_request(last_content):
                response = self.remember_explicit_text(
                    last_content,
                    source_kind="chat.explicit_remember",
                )
                self._log_chat_interaction(raw_input=last_content, response_text=response)
                if record_in_history:
                    self._append_short_term([LLMMessage(role="assistant", content=response)])
                return response

            runtime_mode = getattr(self, "runtime_mode", "ask")
            if runtime_mode == "ask":
                return self._run_chat_response(messages, last_content, record_in_history)
            if runtime_mode == "auto":
                result = self.handle_auto_command(last_content)
                self._log_chat_interaction(raw_input=last_content, response_text=result)
                if record_in_history:
                    self._append_short_term([LLMMessage(role="assistant", content=result)])
                return result

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
                    routing_reason=decision.reason,
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

    def respond_stream(self, messages: list[LLMMessage]) -> Iterator[str | LLMStreamChunk]:
        if not messages:
            self.last_stream_response_raw = "[Пустое сообщение]"
            yield "[Пустое сообщение]"
            return

        last_content = messages[-1].content.strip()
        self._last_user_input = last_content
        record_in_history = self._should_record_in_history(last_content)
        self.last_stream_response_raw = None
        try:
            if record_in_history:
                self._append_short_term(messages)
            self.tracer.log("user_input", last_content)
            self._reset_approval_state()
            self.last_reasoning = None
            self._reset_workspace_diffs()

            if last_content.startswith("/"):
                result = self.handle_tool_command(last_content)
                self.last_stream_response_raw = result
                yield result
                return

            if self.is_explicit_memory_request(last_content):
                response = self.remember_explicit_text(
                    last_content,
                    source_kind="chat.explicit_remember",
                )
                self._log_chat_interaction(raw_input=last_content, response_text=response)
                if record_in_history:
                    self._append_short_term([LLMMessage(role="assistant", content=response)])
                self.last_stream_response_raw = response
                yield response
                return

            runtime_mode = getattr(self, "runtime_mode", "ask")
            if runtime_mode == "ask":
                yield from self._run_chat_response_stream(
                    messages,
                    last_content,
                    record_in_history,
                )
                return
            if runtime_mode == "auto":
                response = self.handle_auto_command(last_content)
                self.last_stream_response_raw = response
                yield response
                return

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
                self.last_stream_response_raw = response
                yield response
                return

            decision_packet = self.decision_handler.evaluate(
                DecisionContext(
                    user_input=last_content,
                    route=decision.route,
                    routing_reason=decision.reason,
                    risk_flags=list(decision.risk_flags),
                    skill_decision=decision.skill_decision,
                ),
            )
            if decision_packet is not None:
                response = self._handle_decision_packet(
                    decision_packet,
                    raw_input=last_content,
                    record_in_history=record_in_history,
                )
                self.last_stream_response_raw = response
                yield response
                return

            if decision.route == "mwv":
                if decision.skill_decision and decision.skill_decision.status == "no_match":
                    self._record_unknown_inbox(last_content, decision)
                    self._record_unknown_skill_candidate(last_content, decision)
                response = self._run_mwv_flow(messages, last_content, decision, record_in_history)
                self.last_stream_response_raw = response
                yield response
                return

            yield from self._run_chat_response_stream(
                messages,
                last_content,
                record_in_history,
            )
        except ApprovalRequired as exc:
            response = self._handle_approval_required(
                exc.request,
                raw_input=last_content,
                record_in_history=record_in_history,
            )
            self.last_stream_response_raw = response
            yield response
        except DecisionRequired as exc:
            response = self._handle_decision_packet(
                exc.packet,
                raw_input=last_content,
                record_in_history=record_in_history,
            )
            self.last_stream_response_raw = response
            yield response
        except Exception as exc:
            self.logger.exception("Agent.respond_stream error: %s", exc)
            self.tracer.log("error", f"Ошибка Agent.respond_stream: {exc}")
            error_text = f"[Ошибка ответа: {exc}]"
            self.last_stream_response_raw = error_text
            yield error_text

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
            web_evidence = self._initial_web_search_evidence()
            messages_with_context, web_evidence = self._prepare_web_search_context(
                last_content,
                messages_with_context,
                web_evidence,
            )
            if (
                web_evidence.requested
                and web_evidence.provider == "local"
                and not web_evidence.executed
            ):
                return self._finalize_chat_response(
                    last_content=last_content,
                    record_in_history=record_in_history,
                    policy_application=policy_application,
                    response_text=self._web_search_not_executed(web_evidence),
                )
            tool_loop_result = self._run_chat_tool_loop_if_available(messages_with_context)
            if tool_loop_result is not None:
                reviewed = self._review_answer(tool_loop_result)
                blocked = self._web_search_block_reason(reviewed, web_evidence)
                if blocked is not None:
                    reviewed = blocked
                self.tracer.log(
                    "reasoning_end",
                    "Ответ получен через native tool loop",
                    {"reply_preview": reviewed[:120]},
                )
                return self._finalize_chat_response(
                    last_content=last_content,
                    record_in_history=record_in_history,
                    policy_application=policy_application,
                    response_text=reviewed,
                )
            reply = self._get_main_brain().generate(messages_with_context)
            web_evidence = self._merge_llm_web_search_evidence(web_evidence, reply)
            reviewed = self._review_answer(reply.text)
            blocked = self._web_search_block_reason(reviewed, web_evidence)
            if blocked is not None:
                reviewed = blocked
            if self.main_config and self.main_config.thinking_enabled:
                self.last_reasoning = reply.reasoning
            self.tracer.log("reasoning_end", "Ответ получен", {"reply_preview": reviewed[:120]})
            return self._finalize_chat_response(
                last_content=last_content,
                record_in_history=record_in_history,
                policy_application=policy_application,
                response_text=reviewed,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error("LLM error: %s", exc)
            self.tracer.log("error", f"Ошибка модели: {exc}")
            error_text = f"[Ошибка модели: {exc}]"
            self._log_chat_interaction(raw_input=last_content, response_text=error_text)
            if record_in_history:
                self._append_short_term([LLMMessage(role="assistant", content=error_text)])
            return error_text

    def _run_chat_tool_loop_if_available(
        self,
        messages: list[LLMMessage],
    ) -> str | None:
        tool_specs = self._chat_read_tool_specs()
        if not tool_specs:
            return None
        result = AgentToolLoop().run(
            brain=self._get_main_brain(),
            gateway=self._build_tool_gateway(safe_mode_override=None),
            messages=messages,
            tools=tool_specs,
            config=self.main_config,
        )
        if not result.tool_calls:
            return result.text
        self.tracer.log(
            "native_tool_loop",
            "chat read-only tool loop executed",
            {
                "tool_calls": len(result.tool_calls),
                "iterations": result.iterations,
                "tools": [item.call.name for item in result.tool_calls],
            },
        )
        return result.text

    def _chat_read_tool_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for name, enabled in self.tool_registry.list_tools().items():
            if not enabled:
                continue
            descriptor = self.tool_registry.get_descriptor(name)
            if descriptor is None or descriptor.capability != "read":
                continue
            if not descriptor.chat_exposed:
                continue
            if not descriptor.description and not descriptor.parameters_schema:
                continue
            specs.append(
                ToolSpec(
                    name=descriptor.name,
                    description=descriptor.description,
                    parameters_schema=dict(descriptor.parameters_schema),
                )
            )
        return specs

    def _run_chat_response_stream(
        self,
        messages: list[LLMMessage],
        last_content: str,
        record_in_history: bool,
    ) -> Iterator[LLMStreamChunk]:
        try:
            self.tracer.log("reasoning_start", "Потоковая генерация ответа моделью")
            policy_application = self._apply_policies(last_content)
            messages_with_context = self._build_context_messages(self.short_term, last_content)
            messages_with_context = self._append_policy_instructions(
                messages_with_context,
                policy_application,
            )
            del messages
            web_evidence = self._initial_web_search_evidence()
            messages_with_context, web_evidence = self._prepare_web_search_context(
                last_content,
                messages_with_context,
                web_evidence,
            )
            if (
                web_evidence.requested
                and web_evidence.provider == "local"
                and not web_evidence.executed
            ):
                blocked_text = self._web_search_not_executed(web_evidence)
                yield LLMStreamChunk(text=blocked_text, mode="append")
                response_text = self._finalize_chat_response(
                    last_content=last_content,
                    record_in_history=record_in_history,
                    policy_application=policy_application,
                    response_text=blocked_text,
                )
                self.last_stream_response_raw = response_text
                return
            collected_text = ""
            brain = self._get_main_brain()
            if web_evidence.requested and web_evidence.provider == "xai_native":
                reply = brain.generate(messages_with_context)
                web_evidence = self._merge_llm_web_search_evidence(web_evidence, reply)
                collected_text = reply.text
                blocked = self._web_search_block_reason(collected_text, web_evidence)
                if blocked is not None:
                    collected_text = blocked
                for idx in range(0, len(collected_text), 80):
                    yield LLMStreamChunk(text=collected_text[idx : idx + 80], mode="append")
            else:
                for chunk in brain.generate_stream_chunks(messages_with_context):
                    if not isinstance(chunk, LLMStreamChunk):
                        continue
                    delta = chunk.text
                    if not delta:
                        continue
                    mode: Literal["append", "replace"] = (
                        "replace" if chunk.mode == "replace" else "append"
                    )
                    if mode == "replace":
                        collected_text = delta
                    else:
                        collected_text = f"{collected_text}{delta}"
                    yield LLMStreamChunk(text=delta, mode=mode, meta=chunk.meta)
            reviewed = self._review_answer(collected_text)
            blocked = self._web_search_block_reason(reviewed, web_evidence)
            if blocked is not None:
                reviewed = blocked
            self.tracer.log("reasoning_end", "Ответ получен", {"reply_preview": reviewed[:120]})
            response_text = self._finalize_chat_response(
                last_content=last_content,
                record_in_history=record_in_history,
                policy_application=policy_application,
                response_text=reviewed,
            )
            self.last_stream_response_raw = response_text
            return
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Stream LLM error: %s", exc)
            self.tracer.log("error", f"Ошибка потоковой модели: {exc}")
            raise

    def _initial_web_search_evidence(self) -> WebSearchEvidence:
        requested = bool(self.main_config and self.main_config.web_search_enabled)
        if not requested:
            evidence = WebSearchEvidence(requested=False, executed=False, provider="none")
            self._log_web_search_evidence(evidence)
            return evidence
        if self.main_config and self.main_config.provider == "xai":
            evidence = WebSearchEvidence(requested=True, executed=False, provider="xai_native")
            self._log_web_search_evidence(evidence)
            return evidence
        evidence = WebSearchEvidence(requested=True, executed=False, provider="local")
        self._log_web_search_evidence(evidence)
        return evidence

    def _prepare_web_search_context(
        self,
        last_content: str,
        messages_with_context: list[LLMMessage],
        evidence: WebSearchEvidence,
    ) -> tuple[list[LLMMessage], WebSearchEvidence]:
        if not evidence.requested or evidence.provider != "local":
            return messages_with_context, evidence
        request = ToolRequest(name="web", args={"query": last_content})
        result = self._call_tool_logged("web_search:runtime", request, safe_mode_override=None)
        if not result.ok:
            failed = WebSearchEvidence(
                requested=True,
                executed=False,
                provider="local",
                tool_call_seen=True,
                error=result.error or "local web search failed",
            )
            self._log_web_search_evidence(failed)
            return messages_with_context, failed
        output_raw = result.data.get("output")
        output = output_raw if isinstance(output_raw, str) else str(result.data)
        local_result_seen = bool(output.strip())
        next_evidence = WebSearchEvidence(
            requested=True,
            executed=local_result_seen,
            provider="local",
            tool_call_seen=True,
            local_result_seen=local_result_seen,
            error=None if local_result_seen else "local web search returned empty result",
        )
        self._log_web_search_evidence(next_evidence)
        if not next_evidence.executed:
            return messages_with_context, next_evidence
        web_context = LLMMessage(
            role="system",
            content=(
                "Verified runtime web search evidence for the next answer:\n"
                f"{output}\n\n"
                "Use only this verified search result as web evidence. "
                "Do not claim additional browsing."
            ),
        )
        return [*messages_with_context, web_context], next_evidence

    def _merge_llm_web_search_evidence(
        self,
        existing: WebSearchEvidence,
        result: LLMResult,
    ) -> WebSearchEvidence:
        if not existing.requested or existing.provider != "xai_native":
            return existing
        if result.web_search_evidence is None:
            merged = WebSearchEvidence(
                requested=True,
                executed=False,
                provider="xai_native",
                error="xAI response contained no web search evidence",
            )
            self._log_web_search_evidence(merged)
            return merged
        self._log_web_search_evidence(result.web_search_evidence)
        return result.web_search_evidence

    def _web_search_block_reason(
        self,
        answer: str,
        evidence: WebSearchEvidence,
    ) -> str | None:
        if evidence.requested and not evidence.executed:
            return self._web_search_not_executed(evidence)
        if self._contains_web_claim(answer) and not evidence.executed:
            return self._web_search_not_executed(
                WebSearchEvidence(
                    requested=evidence.requested,
                    executed=False,
                    provider=evidence.provider,
                    tool_call_seen=evidence.tool_call_seen,
                    citations_count=evidence.citations_count,
                    local_result_seen=evidence.local_result_seen,
                    error="assistant claimed web access without runtime evidence",
                ),
            )
        return None

    def _contains_web_claim(self, answer: str) -> bool:
        normalized = answer.casefold()
        return any(marker in normalized for marker in self._WEB_CLAIM_MARKERS)

    def _web_search_not_executed(self, evidence: WebSearchEvidence) -> str:
        reason = evidence.error or "missing evidence"
        provider = self.main_config.provider if self.main_config else "none"
        return (
            f"web_search_not_executed: {reason}\n"
            f"provider={provider}\n"
            f"mode={evidence.provider}\n"
            f"tool_call_seen={str(evidence.tool_call_seen).lower()}\n"
            f"citations_count={evidence.citations_count}\n"
            f"local_result_seen={str(evidence.local_result_seen).lower()}"
        )

    def _log_web_search_evidence(self, evidence: WebSearchEvidence) -> None:
        provider = self.main_config.provider if self.main_config else "none"
        model = self.main_config.model if self.main_config else "none"
        self.tracer.log(
            "web_search_evidence",
            evidence.provider,
            {
                "provider": provider,
                "model": model,
                "web_required": evidence.requested,
                "web_mode": evidence.provider,
                "endpoint": "xai_responses" if evidence.provider == "xai_native" else "local_tool",
                "tools_sent": ["web_search"] if evidence.provider == "xai_native" else ["web"],
                "tool_call_seen": evidence.tool_call_seen,
                "citations_count": evidence.citations_count,
                "local_result_seen": evidence.local_result_seen,
                "error": evidence.error or "",
            },
        )

    def _finalize_chat_response(
        self,
        *,
        last_content: str,
        record_in_history: bool,
        policy_application: PolicyApplication,
        response_text: str,
    ) -> str:
        final_text = self._append_report_block(
            response_text,
            route="chat",
            trace_id=None,
            attempts=None,
            verifier=None,
            next_steps=None,
            stop_reason_code=None,
            plan_summary="План не требуется для chat-маршрута.",
            execution_summary="Ответ сформирован моделью.",
        )
        self._log_chat_interaction(
            raw_input=last_content,
            response_text=final_text,
            applied_policy_ids=policy_application.applied_policy_ids,
        )
        if record_in_history:
            self._append_short_term([LLMMessage(role="assistant", content=final_text)])
        return final_text

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
