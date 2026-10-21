import { useEffect, useRef, useState } from 'react';

type Evidence = {
  state: 'unknown' | 'supported' | 'unsupported';
  can_check: boolean;
  message?: string;
};

function evidence(value: unknown): Evidence | null {
  if (typeof value !== 'object' || value === null || !('state' in value)
      || !('can_check' in value) || typeof value.can_check !== 'boolean') return null;
  if (value.state !== 'unknown' && value.state !== 'supported' && value.state !== 'unsupported') return null;
  return {
    state: value.state, can_check: value.can_check,
    message: 'message' in value && typeof value.message === 'string' ? value.message : undefined,
  };
}

export function ModelQualification({ sessionId, sessionHeader }: {
  sessionId: string;
  sessionHeader: string;
}) {
  const [snapshot, setSnapshot] = useState<Evidence | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const capabilitiesPath = `/ui/api/sessions/${encodeURIComponent(sessionId)}/model/capabilities`;
  const qualifyPath = `/ui/api/sessions/${encodeURIComponent(sessionId)}/model/qualify`;
  useEffect(() => {
    const pending = new AbortController();
    controller.current = pending;
    void fetch(capabilitiesPath, { headers: { [sessionHeader]: sessionId }, signal: pending.signal })
      .then(async (response) => {
        if (!response.ok) return;
        const result = evidence(await response.json());
        if (!pending.signal.aborted) setSnapshot(result);
      }).catch(() => { /* Unknown остаётся без обещания действия. */ });
    return () => { controller.current?.abort(); };
  }, [capabilitiesPath, sessionHeader, sessionId]);

  const check = async () => {
    controller.current?.abort();
    const pending = new AbortController();
    controller.current = pending;
    setBusy(true); setError(null);
    // Старая положительная проверка отозвана сервером при начале recheck.
    setSnapshot({ state: 'unknown', can_check: true });
    try {
      const response = await fetch(qualifyPath, {
        method: 'POST', headers: { 'Content-Type': 'application/json', [sessionHeader]: sessionId },
        body: JSON.stringify({ confirm: true }), signal: pending.signal,
      });
      if (pending.signal.aborted) return;
      if (!response.ok) throw new Error('Проверка не завершена. Модель занята или контекст изменился.');
      const result = evidence(await response.json());
      if (!pending.signal.aborted) {
        if (!result) throw new Error('Provider не подтвердил результат проверки.');
        setSnapshot(result);
      }
    } catch (cause) {
      if (!pending.signal.aborted) setError(cause instanceof Error ? cause.message : 'Проверка недоступна.');
    } finally {
      if (!pending.signal.aborted) setBusy(false);
    }
  };

  if (!snapshot) return null;
  return <div className="space-y-2 rounded-lg border border-zinc-800 p-3 text-xs" aria-label="Model tool qualification">
    <p className="text-zinc-300">Tools: {busy ? 'проверка…' : snapshot.state === 'supported'
      ? 'проверены для этой сессии' : snapshot.state === 'unsupported' ? 'проверка протокола не пройдена' : 'не проверены'}</p>
    {snapshot.message && <p className="text-zinc-400">{snapshot.message}</p>}
    <p className="text-zinc-500">Проверка отправит четыре коротких запроса выбранной модели. История чата не отправляется, действия на компьютере не выполняются. Разрешения инструментов сохраняются.</p>
    {error && <p role="alert" className="text-amber-400">{error}</p>}
    {snapshot.can_check && <button type="button" disabled={busy} onClick={() => { void check(); }}
      className="rounded border border-zinc-700 px-3 py-1 text-zinc-200 disabled:opacity-50">
      {busy ? 'Проверка…' : snapshot.state === 'supported' ? 'Проверить снова' : 'Проверить tools'}
    </button>}
  </div>;
}
