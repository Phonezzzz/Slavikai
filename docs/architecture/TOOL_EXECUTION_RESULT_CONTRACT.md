# Контракт результата выполнения инструмента

Статусы реализации определяются `docs/runtime_contract_claims.json`. Этот контракт
разделяет authoritative execution outcome и представления результата для потребителей.
Он не объявляет существующий `ToolResult` полным durable canonical result.

## Outcome — implemented для generic process tools

`ToolResult.ok` описывает выполнение запрошенной операции, а не успешный запуск процесса
или доставку ответа. Для `shell`, terminal oneshot и `workspace_run` exit code 0 означает
success; ненулевой код (включая завершение сигналом) означает failure. Tool-specific
семантика допустима только как явный контракт специализированного инструмента.

`ToolResult.failure(error, meta, data=...)` сохраняет доступную диагностику в `data`.
Наличие output не делает failure успешным. Policy/approval остаются authority до execution;
отказ policy не означает, что процесс запускался. Timeout не разрешает автоматический rerun.
Compressor, model response и verifier не могут переписать execution outcome.

HTTP endpoints workspace run/terminal run возвращают отчёт завершившегося процесса с
`ok`, `error`, stdout/stderr и exit code. HTTP 200 означает доставленный отчёт, в том числе
`ok=false`; ошибки до запуска остаются HTTP 400, approval — HTTP 202.

## Полное canonical evidence — target

Полный результат означает сохранение выбранной policy-authorized операции и всех реально
полученных данных до необратимого преобразования. Это не требование неограниченного RAM.

```text
ToolGateway: policy/approval -> execution
                                  |
                     capture bytes / structured result
                                  |
                scoped durable result + immutable payload
                    |             |              |
              audit/replay    UI/verifier    model projection
                                                |
                                     optional compression
                                                |
                                       provider serialization
```

Для subprocess stdout/stderr захватываются раздельно до decode, strip, merge и truncation.
Для PTY фиксируется combined stream и его происхождение. Structured result сохраняет
поля, типы и authoritative metadata. Timeout/cancellation/resource limit сохраняют доступные
partial bytes и явную completeness/capture-error отметку; нельзя выдавать partial за full.
Binary/invalid UTF-8 требуют byte payload с явным encoding представления.

Metadata и большие payload могут храниться отдельно. Publication результата должна
ссылаться только на уже durable payload; незавершённые записи и orphan artifacts очищаются
без уничтожения опубликованных результатов. Storage failure не меняет состоявшийся outcome,
но блокирует обещание recoverability и дальнейшее lossy представление.

Identity связывает result с principal/session/run/tool call и attempt. Она не является
capability: каждое чтение повторно проверяет scope. Lifecycle authority должна использовать
направление `TASK_RUN_LIFECYCLE_CONTRACT.md`, без параллельного compression lifecycle.
Конкретная схема и transaction protocol относятся к следующим foundation PR.

## Потребители и retention — target

UI, model и verifier могут получать разные projections. Audit/replay ссылаются на
canonical evidence; model projection не является source of truth. Retry provider использует
тот же execution result и не повторяет инструмент. Resume/replay различают observation reuse
и новое исполнение с новым attempt. Recovery проходит через ToolGateway/policy и читает
scoped result identity; произвольный filesystem path не допускается.

Default retention полного output — 7 дней **после перехода run в terminal state**.
Retention configurable. Активные, resumable и recovery-pinned результаты GC не удаляет.
Ручное удаление пользователем допускается раньше: остаётся unavailable/tombstone semantics,
без скрытого повторного execution. После restart действуют те же scope и retention правила.
Raw secrets защищаются storage authorization; egress redaction и projection не уничтожают
canonical bytes. Metrics не сохраняют raw sensitive content.

## Readiness gate

До production compression обязательны:

1. Полный canonical result после capture, с честной completeness.
2. Outcome определяется до projection и независимо от compressor.
3. Явная provider-neutral model projection boundary.
4. Стабильное controlled восстановление canonical data.
5. Principal/session/run authorization при каждом чтении.
6. Проверенные restart, retention и deletion semantics.
7. Отсутствие необратимого truncation до canonical capture.

PR A исправляет generic process outcome, но не закрывает остальные gates.
Существующие shell/terminal ограничения output, volatile observations, отсутствие durable
result identity и restart recovery остаются foundation blockers. **TinyJuice readiness:
BLOCKED.** Compression не начинается раньше PR F и успешного прохождения gates.
