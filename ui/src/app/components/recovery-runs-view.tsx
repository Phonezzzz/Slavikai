import { useEffect, useRef, useState } from 'react';

type Run = {
  task_run_id: string;
  recovery_prior_state: string;
  recovery_occurred_at: string;
};

type Operation = {
  operation_id: string;
  tool_attempt_id: string;
  request_path: string;
  publication: string;
  publication_source: string | null;
  publication_snapshot_sha256: string | null;
  publication_snapshot_source: string | null;
  publication_cleanup_warning: boolean | null;
  observation: {
    exit_code: number | null;
    capture_error: string | null;
    timed_out: boolean;
    stdout_bytes_seen: number;
    stdout_bytes_retained: number;
    stdout_eof: boolean;
    stderr_bytes_seen: number;
    stderr_bytes_retained: number;
    stderr_eof: boolean;
  } | null;
};

type Page<T> = { next_cursor: string | null } & T;
type Runs = Page<{ runs: Run[] }>;
type Evidence = Page<{
  recovery_reason: string;
  external_effects: string;
  submission: {
    submission_id: string;
    version: number;
    kind: string;
    content_digest: string;
    has_failure: boolean;
  } | null;
  operations: Operation[];
}>;

export function RecoveryRunsView({ sessionId, sessionHeader }: {
  sessionId: string;
  sessionHeader: string;
}) {
  const [runs, setRuns] = useState<Runs | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => () => pending.current?.abort(), []);

  const read = async (runId?: string, after?: string) => {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true);
    setError(null);
    if (!runId && !after) setRuns(null);
    if (!after) {
      setSelected(runId ?? null);
      setEvidence(null);
    }
    const session = encodeURIComponent(sessionId);
    const path = runId
      ? `/ui/api/sessions/${session}/runs/${encodeURIComponent(runId)}/recovery-evidence`
      : `/ui/api/sessions/${session}/runs/recovery`;
    const query = after ? `?after=${encodeURIComponent(after)}` : '';
    try {
      const response = await fetch(path + query, {
        headers: { [sessionHeader]: sessionId },
        signal: controller.signal,
      });
      if (!response.ok) throw new Error('Recovery unavailable');
      const body = await response.json();
      if (controller.signal.aborted) return;
      if (runId) {
        const page = body as Evidence;
        setEvidence((previous) => after && previous
          ? { ...page, operations: [...previous.operations, ...page.operations] }
          : page);
      } else {
        const page = body as Runs;
        setRuns((previous) => after && previous
          ? { ...page, runs: [...previous.runs, ...page.runs] }
          : page);
      }
    } catch {
      if (!controller.signal.aborted) setError('Не удалось прочитать данные для сверки.');
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  };

  return (
    <section className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <h3 className="text-sm font-medium text-zinc-200">Сверка запусков</h3>
      <button type="button" className="text-xs underline" onClick={() => void read()}>
        Показать запуски для сверки
      </button>
      {loading && <p role="status">Чтение данных…</p>}
      {error && <p role="alert">{error}</p>}
      {runs && !runs.runs.length && <p>В этой сессии нет запусков, ожидающих сверки.</p>}
      {runs && (
        <ul className="space-y-2">
          {runs.runs.map((run) => (
            <li key={run.task_run_id}>
              <button
                type="button"
                className="break-all text-left text-xs underline"
                aria-pressed={selected === run.task_run_id}
                onClick={() => void read(run.task_run_id)}
              >
                {run.task_run_id} · {run.recovery_prior_state} · {run.recovery_occurred_at}
              </button>
            </li>
          ))}
        </ul>
      )}
      {runs?.next_cursor && (
        <button type="button" className="text-xs underline" disabled={loading} onClick={() => void read(undefined, runs.next_cursor ?? undefined)}>Следующая страница запусков</button>
      )}
      {evidence && selected && (
        <div className="space-y-2 break-all text-xs">
          <h4>Свидетельства запуска {selected}</h4>
          <p>Причина перехода к сверке: {evidence.recovery_reason === 'server_restart_requires_reconciliation'
            ? 'Сервер перезапустился до завершения запуска.'
            : evidence.recovery_reason}</p>
          <div className="space-y-1">
            <h5>Сохранённый результат</h5>
            {evidence.submission ? (
              <>
                <p>ID результата: {evidence.submission.submission_id}</p>
                <p>Версия: {evidence.submission.version} · Источник: {evidence.submission.kind}</p>
                <p>SHA-256 результата: {evidence.submission.content_digest}</p>
                <p>{evidence.submission.has_failure
                  ? 'В результате отмечен сбой.'
                  : 'В результате сбой не отмечен.'}</p>
                <p>Запись подтверждает отправку результата. Приёмка требует отдельного подтверждения.</p>
              </>
            ) : (
              <p>Сохранённого результата для сверки нет.</p>
            )}
          </div>
          <p>Внешние эффекты: {evidence.external_effects}</p>
          {!evidence.operations.length && (
            <p>Сохранённых попыток нет. Внешние эффекты всё ещё требуют сверки.</p>
          )}
          <ul className="space-y-2">
            {evidence.operations.map((operation) => (
              <li key={operation.operation_id}>
                <p>{operation.request_path} · {operation.tool_attempt_id}</p>
                <p>Публикация: {operation.publication} · {operation.publication_source ?? 'источник отсутствует'}</p>
                {operation.publication_snapshot_sha256 && (
                  <p>SHA-256: {operation.publication_snapshot_sha256} · {operation.publication_snapshot_source}</p>
                )}
                <p>{operation.observation
                  ? `Код выхода: ${operation.observation.exit_code ?? 'неизвестен'}${operation.observation.capture_error ? ` · ${operation.observation.capture_error}` : ''}`
                  : 'Наблюдение отсутствует'}</p>
                {operation.publication_cleanup_warning === true && (
                  <p>После публикации очистка завершилась с ошибкой.</p>
                )}
                {operation.observation && (
                  <>
                    {operation.observation.timed_out && <p>Время выполнения истекло.</p>}
                    <p>stdout: сохранено {operation.observation.stdout_bytes_retained} из {operation.observation.stdout_bytes_seen} прочитанных байтов · {operation.observation.stdout_eof ? 'EOF достигнут' : 'EOF не достигнут'}</p>
                    <p>stderr: сохранено {operation.observation.stderr_bytes_retained} из {operation.observation.stderr_bytes_seen} прочитанных байтов · {operation.observation.stderr_eof ? 'EOF достигнут' : 'EOF не достигнут'}</p>
                  </>
                )}
              </li>
            ))}
          </ul>
          {evidence.next_cursor && (
            <button type="button" className="underline" disabled={loading} onClick={() => void read(selected, evidence.next_cursor ?? undefined)}>
              Следующая страница свидетельств
            </button>
          )}
        </div>
      )}
    </section>
  );
}
