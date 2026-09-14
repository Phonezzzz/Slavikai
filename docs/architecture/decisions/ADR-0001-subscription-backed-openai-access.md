# ADR-0001 — Subscription-backed OpenAI/ChatGPT access is a Target capability

- **Status:** accepted
- **Date:** 2026-09-15
- **Decision owner:** SlavikAI product/architecture owner
- **Affected domains:** model/provider access, authentication/credentials, security/trust,
  routing/fallback, observability, verification, user interaction

## Context

SlavikAI должен уметь использовать доступ пользователя к моделям OpenAI, оплаченный через
его ChatGPT/OpenAI account и consumer subscription, а не требовать для этого сценария только
отдельный API-billed key. Subscription-backed access и OpenAI API имеют разные entitlement,
billing, authentication, quota, supported interfaces и trust boundaries. Поэтому один нельзя
неявно подменять другим.

Предыдущая документация сохраняла эту тему только как research scope. Это не отражало
продуктовое решение владельца: capability нужна в конечном SlavikAI; исследованию подлежит
безопасный и поддерживаемый способ её реализации.

## Decision

1. Target SlavikAI обязательно включает first-class, user-selectable
   **subscription-backed OpenAI/ChatGPT access route**.
2. Этот route имеет отдельную identity от `official_api`/API-key route. Его нельзя считать
   API credit, автоматически переводить на API billing или маскировать под обычный provider
   endpoint.
3. Route может считаться usable только при подтверждённых authentication, account ownership,
   entitlement, quota/readiness и поддерживаемом/разрешённом interface для точного продукта и
   времени проверки.
4. Выбор route и любой fallback должны быть observable и policy-controlled. Запрещён silent
   переход между subscription, API billing, другим account, provider, trust/privacy или
   data-egress boundary.
5. Если поддерживаемого и разрешённого access path нет или entitlement/readiness не
   подтверждены, runtime обязан сообщить `unavailable`/`blocked`, а не обходить ограничение
   через скрытую автоматизацию Web UI, чужие credentials или неподтверждённый proxy.
6. Credentials/session material являются principal-scoped sensitive state: raw tokens,
   cookies и secrets не хранятся в обычной model configuration и не попадают в logs,
   exports, prompts или diagnostics.

## Deliberately unresolved implementation decisions

ADR принимает обязательность capability, но не выдумывает ещё не проверенный механизм:

- конкретный official auth/authorization flow;
- transport/API/SDK и допустимость account/session integration;
- необходимость и допустимость `web_session` как Access Mode;
- entitlement и quota discovery/revalidation;
- model catalog, capability qualification и freshness;
- session recovery, revocation, challenge и multi-account semantics;
- concrete UI, persistence schema и rollout.

Эти вопросы требуют authoritative product/security research и отдельного design/ADR. Они не
могут отменить Target-требование молча; если requirement станет невыполнимым, пересмотр должен
явно supersede этот ADR с evidence и последствиями.

## Alternatives considered

- **API-key access only:** rejected; не выполняет выбранный subscription-backed user outcome.
- **Treat subscription as API entitlement:** rejected; это разные billing/auth products.
- **Mandate an unofficial Web-session proxy now:** rejected; техническая демонстрация не
  доказывает permitted, secure или durable production contract.
- **Accept Target capability and gate implementation on verified compliant routes:** selected;
  сохраняет продуктовую цель без ложного утверждения о текущей реализации.

## Consequences

- Capability Discovery обязана включить model/provider access и subscription/account access в
  системную карту; точная domain decomposition остаётся открытой.
- Provider/model registry должен в будущем различать model family, Access Mode, auth identity,
  upstream transport, protocol adapter, capability provenance/qualification и trust policy.
- Runtime не может считать текущий DeepSeek multimodal prototype реализацией этого решения.
- Consolidation должна сохранить этот ADR и связанные research findings, но не обязана
  переносить экспериментальный provider-specific код в production descendant.
