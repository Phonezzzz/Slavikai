from __future__ import annotations

# ruff: noqa: F401
import asyncio
from collections.abc import Callable, Generator, Iterator
from typing import TYPE_CHECKING, Literal

from core.agent_response import AgentResponse, ResponseFailure, ResponseProduced
from core.approval_policy import ApprovalRequired
from core.decision.handler import DecisionContext
from core.decision.memory_save import build_memory_save_packet
from core.mwv.models import StopReasonCode
from core.mwv.routing import RouteDecision, classify_request
from core.skills.index import SkillMatchDecision
from core.tool_loop import AgentToolLoop, AgentToolLoopResult, ToolLoopExecutionError
from llm.cancellation import GenerationCancelled, cancellation_requested
from llm.retry import visible_provider_error
from llm.stream_model import Done, Error, StreamEvent, TextDelta, Usage
from llm.types import LLMResult, ToolSpec, WebSearchEvidence
from shared.models import LLMMessage, ToolRequest, ToolResult

if TYPE_CHECKING:
    import logging

    from config.memory_config import MemoryConfig
    from core.approval_policy import ApprovalRequest
    from core.auto_runtime import AutoRunOutcome
    from core.decision.handler import DecisionHandler
    from core.decision.models import DecisionPacket
    from core.desktop_runtime import DesktopRunOutcome, DesktopRuntime
    from core.mwv.models import VerificationResult
    from core.rule_engine import PolicyApplication
    from core.skills.index import SkillIndex, SkillMatch, SkillResolution
    from core.tool_gateway import ToolGateway
    from core.tracer import Tracer
    from llm.brain_base import Brain
    from llm.types import ModelConfig
    from shared.models import JSONValue
    from tools.tool_registry import ToolRegistry


def _text_response_events(
    text: str, *, runtime_result: AutoRunOutcome | None = None
) -> Iterator[StreamEvent]:
    if text:
        yield TextDelta(text=text)
    yield ResponseProduced(AgentResponse(text, runtime_result=runtime_result))
    yield Done()


def _tool_stream_observations(
    events: Generator[StreamEvent, None, AgentToolLoopResult],
) -> Generator[StreamEvent, None, AgentToolLoopResult]:
    """Tool-loop Done не завершает внешний Agent stream до response projection."""
    try:
        while True:
            try:
                event = next(events)
            except StopIteration as stopped:
                if not isinstance(stopped.value, AgentToolLoopResult):
                    raise TypeError("tool stream requires AgentToolLoopResult") from stopped
                return stopped.value
            if not isinstance(event, Done):
                yield event
    finally:
        events.close()


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
        desktop_runtime: DesktopRuntime
        last_plan_summary: str | None
        last_execution_summary: str | None

        def _should_record_in_history(self, user_input: str) -> bool: ...
        def _append_short_term(
            self,
            messages: list[LLMMessage],
            *,
            history: list[LLMMessage] | None = None,
        ) -> None: ...
        def _reset_approval_state(self, *, cancel_runtime: bool = True) -> None: ...
        def _capture_chat_approval(
            self,
            exc: ApprovalRequired,
            raw_input: str,
            *,
            policy_application: PolicyApplication | None = None,
            web_evidence: WebSearchEvidence | None = None,
            record_in_history: bool = False,
        ) -> None: ...

        last_approval_source_endpoint: str | None
        last_approval_resume_payload: dict[str, JSONValue] | None

        def _reset_workspace_diffs(self) -> None: ...
        def handle_tool_command(self, command: str) -> str: ...
        def handle_auto_command(
            self,
            goal: str,
            *,
            skill_resolution: SkillResolution | None = None,
        ) -> AutoRunOutcome: ...
        def is_explicit_memory_request(self, text: str) -> bool: ...
        def build_memory_save_preview(
            self,
            text: str,
            *,
            source_kind: str,
            source_id: str | None = None,
            lang_hint: str | None = None,
        ) -> dict[str, JSONValue]: ...
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
            confirmed_decision: bool = False,
        ) -> ToolResult: ...
        def _build_tool_gateway(
            self,
            *,
            pre_call: Callable[[ToolRequest], object | None] | None = None,
            post_call: Callable[[ToolRequest, ToolResult, object | None], None] | None = None,
            safe_mode_override: bool | None = None,
            confirmed_decision: bool = False,
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
            skill: dict[str, JSONValue] | None = None,
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
            skill: dict[str, JSONValue] | None = None,
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

    def respond(self, messages: list[LLMMessage]) -> AgentResponse:
        if not messages:
            return AgentResponse("[Пустое сообщение]")

        last_content = messages[-1].content.strip()
        self._last_user_input = last_content
        record_in_history = self._should_record_in_history(last_content)
        outcome: AutoRunOutcome | None = None
        try:
            if record_in_history:
                self._append_short_term(messages)
            self.tracer.log("user_input", last_content)
            self._reset_approval_state()
            self.last_reasoning = None
            self._reset_workspace_diffs()

            if last_content.startswith("/"):
                return AgentResponse(self.handle_tool_command(last_content))

            runtime_mode = getattr(self, "runtime_mode", "ask")
            if self.is_explicit_memory_request(last_content):
                preview = self.build_memory_save_preview(
                    last_content,
                    source_kind="chat.explicit_remember",
                )
                claims = preview.get("claims")
                if isinstance(claims, list) and claims:
                    return AgentResponse(
                        self._handle_decision_packet(
                            build_memory_save_packet(preview),
                            raw_input=last_content,
                            record_in_history=record_in_history,
                        )
                    )
                response = "Не удалось выделить изменения для Memory."
                self._log_chat_interaction(raw_input=last_content, response_text=response)
                if record_in_history:
                    self._append_short_term([LLMMessage(role="assistant", content=response)])
                return AgentResponse(response)

            if runtime_mode == "ask":
                return self._run_chat_response(messages, last_content, record_in_history)
            if runtime_mode == "auto":
                skill_decision, skill_resolution = self._resolve_skill_run(last_content)
                if skill_decision and skill_decision.status == "deprecated":
                    response = self._format_skill_block(skill_decision)
                    self._log_chat_interaction(raw_input=last_content, response_text=response)
                    if record_in_history:
                        self._append_short_term([LLMMessage(role="assistant", content=response)])
                    return AgentResponse(response)
                decision_packet = self.decision_handler.evaluate(
                    DecisionContext(
                        user_input=last_content,
                        route="auto",
                        routing_reason="auto_skill_match",
                        skill_decision=skill_decision,
                    ),
                )
                if decision_packet is not None:
                    return AgentResponse(
                        self._handle_decision_packet(
                            decision_packet,
                            raw_input=last_content,
                            record_in_history=record_in_history,
                        )
                    )
                outcome = self.handle_auto_command(
                    last_content,
                    skill_resolution=skill_resolution,
                )
                result = outcome.text
                self._log_chat_interaction(raw_input=last_content, response_text=result)
                if record_in_history:
                    self._append_short_term([LLMMessage(role="assistant", content=result)])
                return AgentResponse(result, runtime_result=outcome)
            if runtime_mode == "desktop":
                return AgentResponse(self._run_desktop_response(last_content, record_in_history))

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
                self._log_chat_interaction(raw_input=last_content, response_text=response)
                if record_in_history:
                    self._append_short_term([LLMMessage(role="assistant", content=response)])
                return AgentResponse(response)
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
                return AgentResponse(
                    self._handle_decision_packet(
                        decision_packet,
                        raw_input=last_content,
                        record_in_history=record_in_history,
                    )
                )
            if decision.route == "mwv":
                if decision.skill_decision and decision.skill_decision.status == "no_match":
                    self._record_unknown_inbox(last_content, decision)
                    self._record_unknown_skill_candidate(last_content, decision)
                return AgentResponse(
                    self._run_mwv_flow(messages, last_content, decision, record_in_history)
                )
            return self._run_chat_response(messages, last_content, record_in_history)
        except ApprovalRequired as exc:
            self._capture_chat_approval(exc, last_content, record_in_history=record_in_history)
            return AgentResponse(
                self._handle_approval_required(
                    exc.request,
                    source_endpoint=self.last_approval_source_endpoint,
                    resume_payload=self.last_approval_resume_payload,
                    raw_input=last_content,
                    record_in_history=record_in_history,
                )
            )
        except Exception as exc:
            self.logger.exception("Agent.respond error: %s", exc)
            self.tracer.log("error", f"Ошибка Agent.respond: {exc}")
            error_code, visible_message, _ = visible_provider_error(exc)
            error_text = f"[{visible_message}]"
            try:
                self._log_chat_interaction(raw_input=last_content, response_text=error_text)
            except Exception as log_exc:  # noqa: BLE001
                self.logger.error("Ошибка записи InteractionLog: %s", log_exc)
            if record_in_history:
                self._append_short_term([LLMMessage(role="assistant", content=error_text)])
            return AgentResponse(error_text, outcome, ResponseFailure(error_code, visible_message))

    def respond_stream(
        self,
        messages: list[LLMMessage],
        cancellation_token: asyncio.Event | None = None,
    ) -> Iterator[StreamEvent]:
        if cancellation_requested(cancellation_token):
            yield Done(finish_reason="cancelled")
            return
        if not messages:
            self.last_stream_response_raw = "[Пустое сообщение]"
            yield from _text_response_events("[Пустое сообщение]")
            return

        last_content = messages[-1].content.strip()
        self._last_user_input = last_content
        record_in_history = self._should_record_in_history(last_content)
        self.last_stream_response_raw = None
        outcome: AutoRunOutcome | None = None
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
                yield from _text_response_events(result)
                return

            runtime_mode = getattr(self, "runtime_mode", "ask")
            if self.is_explicit_memory_request(last_content):
                preview = self.build_memory_save_preview(
                    last_content,
                    source_kind="chat.explicit_remember",
                )
                claims = preview.get("claims")
                if isinstance(claims, list) and claims:
                    response = self._handle_decision_packet(
                        build_memory_save_packet(preview),
                        raw_input=last_content,
                        record_in_history=record_in_history,
                    )
                else:
                    response = "Не удалось выделить изменения для Memory."
                    self._log_chat_interaction(raw_input=last_content, response_text=response)
                    if record_in_history:
                        self._append_short_term([LLMMessage(role="assistant", content=response)])
                self.last_stream_response_raw = response
                yield from _text_response_events(response)
                return

            if runtime_mode == "ask":
                yield from self._run_chat_response_stream(
                    messages,
                    last_content,
                    record_in_history,
                    cancellation_token,
                )
                return
            if runtime_mode == "auto":
                skill_decision, skill_resolution = self._resolve_skill_run(last_content)
                if skill_decision and skill_decision.status == "deprecated":
                    response = self._format_skill_block(skill_decision)
                    self.last_stream_response_raw = response
                    yield from _text_response_events(response)
                    return
                decision_packet = self.decision_handler.evaluate(
                    DecisionContext(
                        user_input=last_content,
                        route="auto",
                        routing_reason="auto_skill_match",
                        skill_decision=skill_decision,
                    ),
                )
                if decision_packet is not None:
                    response = self._handle_decision_packet(
                        decision_packet,
                        raw_input=last_content,
                        record_in_history=record_in_history,
                    )
                    self.last_stream_response_raw = response
                    yield from _text_response_events(response)
                    return
                outcome = self.handle_auto_command(
                    last_content,
                    skill_resolution=skill_resolution,
                )
                response = outcome.text
                self.last_stream_response_raw = response
                yield from _text_response_events(response, runtime_result=outcome)
                return
            if runtime_mode == "desktop":
                response = self._run_desktop_response(
                    last_content,
                    record_in_history,
                    cancellation_token=cancellation_token,
                )
                self.last_stream_response_raw = response
                yield from _text_response_events(response)
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
                yield from _text_response_events(response)
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
                yield from _text_response_events(response)
                return

            if decision.route == "mwv":
                if decision.skill_decision and decision.skill_decision.status == "no_match":
                    self._record_unknown_inbox(last_content, decision)
                    self._record_unknown_skill_candidate(last_content, decision)
                response = self._run_mwv_flow(messages, last_content, decision, record_in_history)
                self.last_stream_response_raw = response
                yield from _text_response_events(response)
                return

            yield from self._run_chat_response_stream(
                messages,
                last_content,
                record_in_history,
                cancellation_token,
            )
        except ApprovalRequired as exc:
            self._capture_chat_approval(exc, last_content, record_in_history=record_in_history)
            response = self._handle_approval_required(
                exc.request,
                source_endpoint=self.last_approval_source_endpoint,
                resume_payload=self.last_approval_resume_payload,
                raw_input=last_content,
                record_in_history=record_in_history,
            )
            self.last_stream_response_raw = response
            yield from _text_response_events(response)
        except Exception as exc:
            self.logger.exception("Agent.respond_stream error: %s", exc)
            self.tracer.log("error", f"Ошибка Agent.respond_stream: {exc}")
            error_code, visible_message, _ = visible_provider_error(exc)
            error_text = f"[{visible_message}]"
            self.last_stream_response_raw = error_text
            yield Error(message=visible_message, code=error_code)
            yield ResponseProduced(
                AgentResponse(error_text, outcome, ResponseFailure(error_code, visible_message))
            )
            yield Done(finish_reason="error")

    def _run_chat_response(
        self,
        messages: list[LLMMessage],
        last_content: str,
        record_in_history: bool,
    ) -> AgentResponse:
        runtime_result: AgentToolLoopResult | LLMResult | None = None
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
            tool_loop_result = self._run_chat_tool_loop_if_available(messages_with_context)
            runtime_result = tool_loop_result
            if tool_loop_result is not None:
                if tool_loop_result.tool_calls:
                    self.tracer.log(
                        "native_tool_loop",
                        "chat read-only tool loop executed",
                        {
                            "tool_calls": len(tool_loop_result.tool_calls),
                            "iterations": tool_loop_result.iterations,
                            "tools": [item.call.name for item in tool_loop_result.tool_calls],
                        },
                    )
                web_evidence = self._merge_tool_web_search_evidence(web_evidence, tool_loop_result)
                reviewed = self._review_answer(tool_loop_result.text)
                blocked = self._web_search_block_reason(reviewed, web_evidence)
                if blocked is not None:
                    reviewed = blocked
                self.tracer.log(
                    "reasoning_end",
                    "Ответ получен через native tool loop",
                    {"reply_preview": reviewed[:120]},
                )
                return AgentResponse(
                    self._finalize_chat_response(
                        last_content=last_content,
                        record_in_history=record_in_history,
                        policy_application=policy_application,
                        response_text=reviewed,
                    ),
                    runtime_result=runtime_result,
                )
            reply = self._get_main_brain().generate(messages_with_context)
            runtime_result = reply
            web_evidence = self._merge_llm_web_search_evidence(web_evidence, reply)
            reviewed = self._review_answer(reply.text)
            blocked = self._web_search_block_reason(reviewed, web_evidence)
            if blocked is not None:
                reviewed = blocked
            if self.main_config and self.main_config.thinking_enabled:
                self.last_reasoning = reply.reasoning
            self.tracer.log("reasoning_end", "Ответ получен", {"reply_preview": reviewed[:120]})
            return AgentResponse(
                self._finalize_chat_response(
                    last_content=last_content,
                    record_in_history=record_in_history,
                    policy_application=policy_application,
                    response_text=reviewed,
                ),
                runtime_result=runtime_result,
            )
        except ApprovalRequired as exc:
            self._capture_chat_approval(
                exc,
                last_content,
                policy_application=policy_application,
                web_evidence=web_evidence,
                record_in_history=record_in_history,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, ToolLoopExecutionError):
                runtime_result = exc.result
                exc = exc.cause
            self.logger.error("LLM error: %s", exc)
            try:
                self.tracer.log("error", f"Ошибка модели: {exc}")
            except Exception as trace_exc:  # noqa: BLE001
                self.logger.error("Ошибка записи failed response trace: %s", trace_exc)
            error_text = f"[Ошибка модели: {exc}]"
            try:
                self._log_chat_interaction(raw_input=last_content, response_text=error_text)
                if record_in_history:
                    self._append_short_term([LLMMessage(role="assistant", content=error_text)])
            except Exception as log_exc:  # noqa: BLE001
                self.logger.error("Ошибка записи failed response: %s", log_exc)
            code, message, _ = visible_provider_error(exc)
            return AgentResponse(error_text, runtime_result, ResponseFailure(code, message))

    def _run_desktop_response(
        self,
        goal: str,
        record_in_history: bool,
        *,
        cancellation_token: asyncio.Event | None = None,
    ) -> str:
        outcome = self.desktop_runtime.run(goal, cancellation_token=cancellation_token)
        return self._finalize_desktop_response(outcome, goal, record_in_history)

    def _finalize_desktop_response(
        self, outcome: DesktopRunOutcome, goal: str, record_in_history: bool
    ) -> str:
        response = self._project_desktop_outcome(outcome)
        self._log_chat_interaction(raw_input=goal, response_text=response)
        if record_in_history:
            self._append_short_term([LLMMessage(role="assistant", content=response)])
        return response

    def _project_desktop_outcome(self, outcome: DesktopRunOutcome) -> str:
        self.last_plan_summary = "Desktop использовал native tool loop для host execution."
        self.last_execution_summary = (
            f"tool_calls={len(outcome.loop_result.tool_calls)}, "
            f"iterations={outcome.loop_result.iterations}, "
            f"verification={outcome.verification.status.value}"
        )
        response = self._append_report_block(
            outcome.text,
            route="desktop",
            trace_id=None,
            attempts=(1, 1),
            verifier=outcome.verification,
            next_steps=[],
            stop_reason_code=(
                None
                if outcome.verification.ok and outcome.loop_result.error is None
                else StopReasonCode.VERIFIER_FAILED
            ),
            plan_summary=self.last_plan_summary,
            execution_summary=self.last_execution_summary,
        )
        return response

    def _run_chat_tool_loop_if_available(
        self,
        messages: list[LLMMessage],
    ) -> AgentToolLoopResult | None:
        brain = self._get_main_brain()
        if not brain.supports_native_tools:
            return None
        tool_specs = self._chat_read_tool_specs()
        if not tool_specs:
            return None
        result = AgentToolLoop().run(
            brain=brain,
            gateway=self._build_tool_gateway(safe_mode_override=None),
            messages=messages,
            tools=tool_specs,
            config=self.main_config,
        )
        return result

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
        cancellation_token: asyncio.Event | None = None,
    ) -> Iterator[StreamEvent]:
        runtime_result: AgentToolLoopResult | LLMResult | None = None
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
            collected_text = ""
            brain = self._get_main_brain()
            if web_evidence.requested and web_evidence.provider == "xai_native":
                if cancellation_requested(cancellation_token):
                    yield Done(finish_reason="cancelled")
                    return
                generate_cancellable = getattr(brain, "generate_cancellable", None)
                try:
                    if cancellation_token is not None and callable(generate_cancellable):
                        reply = generate_cancellable(
                            messages_with_context,
                            config=self.main_config,
                            cancellation_token=cancellation_token,
                        )
                    else:
                        reply = brain.generate(messages_with_context)
                except GenerationCancelled:
                    yield Done(finish_reason="cancelled")
                    self.last_stream_response_raw = None
                    return
                runtime_result = reply
                if cancellation_requested(cancellation_token):
                    yield ResponseProduced(AgentResponse("", runtime_result))
                    yield Done(finish_reason="cancelled")
                    return
                web_evidence = self._merge_llm_web_search_evidence(web_evidence, reply)
                collected_text = reply.text
                blocked = self._web_search_block_reason(collected_text, web_evidence)
                if blocked is not None:
                    collected_text = blocked
                for idx in range(0, len(collected_text), 80):
                    yield TextDelta(text=collected_text[idx : idx + 80])
                if reply.usage is not None:
                    yield Usage(usage=reply.usage)
            else:
                tool_specs = self._chat_read_tool_specs() if brain.supports_streaming_tools else []
                tool_loop_result = yield from _tool_stream_observations(
                    AgentToolLoop().run_stream_events(
                        brain=brain,
                        gateway=self._build_tool_gateway(safe_mode_override=None),
                        messages=messages_with_context,
                        tools=tool_specs,
                        config=self.main_config,
                        cancellation_token=cancellation_token,
                    )
                )
                runtime_result = tool_loop_result
                if tool_loop_result.cancelled:
                    self.last_stream_response_raw = None
                    yield ResponseProduced(AgentResponse("", runtime_result))
                    yield Done(finish_reason="cancelled")
                    return
                web_evidence = self._merge_tool_web_search_evidence(web_evidence, tool_loop_result)
                collected_text = tool_loop_result.text
                if tool_loop_result.tool_calls:
                    self.tracer.log(
                        "native_tool_loop",
                        "streaming chat read-only tool loop executed",
                        {
                            "tool_calls": len(tool_loop_result.tool_calls),
                            "iterations": tool_loop_result.iterations,
                            "tools": [item.call.name for item in tool_loop_result.tool_calls],
                        },
                    )
                if tool_loop_result.error is not None:
                    error_text = f"[Ошибка модели: {tool_loop_result.error}]"
                    self._log_chat_interaction(
                        raw_input=last_content,
                        response_text=error_text,
                    )
                    self.last_stream_response_raw = error_text
                    yield ResponseProduced(AgentResponse(error_text, runtime_result))
                    yield Done(finish_reason="error")
                    return
            if cancellation_requested(cancellation_token):
                yield ResponseProduced(AgentResponse("", runtime_result))
                yield Done(finish_reason="cancelled")
                self.last_stream_response_raw = None
                return
            reviewed = self._review_answer(collected_text)
            blocked = self._web_search_block_reason(reviewed, web_evidence)
            if blocked is not None:
                reviewed = blocked
            if cancellation_requested(cancellation_token):
                yield ResponseProduced(AgentResponse("", runtime_result))
                yield Done(finish_reason="cancelled")
                self.last_stream_response_raw = None
                return
            self.tracer.log("reasoning_end", "Ответ получен", {"reply_preview": reviewed[:120]})
            response_text = self._finalize_chat_response(
                last_content=last_content,
                record_in_history=record_in_history,
                policy_application=policy_application,
                response_text=reviewed,
            )
            self.last_stream_response_raw = response_text
            yield ResponseProduced(AgentResponse(response_text, runtime_result))
            yield Done()
            return
        except ApprovalRequired as exc:
            self._capture_chat_approval(
                exc,
                last_content,
                policy_application=policy_application,
                web_evidence=web_evidence,
                record_in_history=record_in_history,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Stream LLM error: %s", exc)
            try:
                self.tracer.log("error", f"Ошибка потоковой модели: {exc}")
            except Exception as trace_exc:  # noqa: BLE001
                self.logger.error("Ошибка записи failed stream trace: %s", trace_exc)
            code, message, _ = visible_provider_error(exc)
            error_text = f"[{message}]"
            self.last_stream_response_raw = error_text
            yield Error(message=message, code=code)
            yield ResponseProduced(
                AgentResponse(error_text, runtime_result, ResponseFailure(code, message))
            )
            yield Done(finish_reason="error")

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
        return [
            *messages_with_context,
            LLMMessage(
                role="system",
                content=(
                    "Web search was explicitly enabled for this request. Use the available web "
                    "tool through native tool calling before answering. Only successful tool "
                    "results are web evidence; never claim browsing without such evidence."
                ),
            ),
        ], evidence

    def _merge_tool_web_search_evidence(
        self, existing: WebSearchEvidence, result: AgentToolLoopResult
    ) -> WebSearchEvidence:
        if existing.provider == "xai_native":
            return existing
        web_calls = [item for item in result.tool_calls if item.call.name == "web"]
        if not web_calls:
            return existing
        successful = any(
            item.result.ok
            and isinstance(output := item.result.data.get("output"), str)
            and bool(output.strip())
            for item in web_calls
        )
        evidence = WebSearchEvidence(
            requested=existing.requested,
            executed=successful,
            provider="local",
            tool_call_seen=True,
            local_result_seen=successful,
            error=None if successful else web_calls[-1].result.error or "empty web result",
        )
        self._log_web_search_evidence(evidence)
        return evidence

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

    def _resolve_skill_run(
        self,
        user_input: str,
    ) -> tuple[SkillMatchDecision | None, SkillResolution | None]:
        if self.skill_index is None:
            self._apply_skill_decision(None)
            return None, None
        decision = self.skill_index.match_decision(user_input)
        self._apply_skill_decision(decision)
        if decision.status != "matched" or decision.match is None:
            return decision, None
        return decision, self.skill_index.resolve_match(decision.match)

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
                skill=self._blocked_skill_report(decision),
            )
        return "Навык не может быть применен."

    def _blocked_skill_report(
        self,
        decision: SkillMatchDecision,
    ) -> dict[str, JSONValue]:
        if decision.status == "deprecated" and decision.match is not None:
            return {
                "status": "skipped",
                "skill_id": decision.match.entry.id,
                "version": decision.match.entry.version,
                "supporting_skills": [],
                "reason": "deprecated",
            }
        return {
            "status": "skipped",
            "skill_id": None,
            "version": None,
            "supporting_skills": [],
            "reason": "unknown",
        }
