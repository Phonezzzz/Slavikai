from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from core.mwv.models import MWVMessage, RunContext, TaskPacket
from core.mwv.routing import MessageLike, RouteDecision, classify_request
from shared.models import JSONValue

RouteClassifier = Callable[[Sequence[MessageLike], str, dict[str, JSONValue] | None], RouteDecision]
TaskBuilder = Callable[[Sequence[MWVMessage], RunContext], TaskPacket]


@dataclass(frozen=True)
class ManagerRuntime:
    task_builder: TaskBuilder
    route_classifier: RouteClassifier = classify_request

    def decide_route(
        self,
        messages: Sequence[MessageLike],
        user_input: str,
        context: dict[str, JSONValue] | None = None,
    ) -> RouteDecision:
        return self.route_classifier(messages, user_input, context)

    def build_task_packet(self, messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
        return self.task_builder(messages, context)
