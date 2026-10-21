import { useEffect, useRef, useState } from 'react';

import { extractErrorMessage } from '../session-payload';

type Snapshot = {
  provider: 'local' | 'openai';
  model: string;
  state: 'missing' | 'package_missing' | 'downloading' | 'ready' | 'error';
  canDownload: boolean;
};

function parseSnapshot(payload: unknown): Snapshot {
  if (!payload || typeof payload !== 'object'
    || !('provider' in payload) || !('model' in payload) || !('state' in payload)) {
    throw new Error('Incomplete embedding status.');
  }
  const { provider, model, state } = payload;
  if ((provider !== 'local' && provider !== 'openai') || typeof model !== 'string' || !model.trim()
    || (state !== 'missing' && state !== 'package_missing' && state !== 'downloading'
      && state !== 'ready' && state !== 'error')) {
    throw new Error('Unknown embedding status.');
  }
  return { provider, model, state, canDownload: 'can_download' in payload && payload.can_download === true };
}

export function EmbeddingNotice() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [busy, setBusy] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const read = useRef<AbortController | null>(null);
  const write = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    read.current = controller;
    void (async () => {
      try {
        const response = await fetch('/ui/api/embeddings/status', { signal: controller.signal });
        const payload: unknown = await response.json();
        if (controller.signal.aborted) return;
        if (!response.ok) throw new Error(extractErrorMessage(payload, 'Unable to check semantic search.'));
        setSnapshot(parseSnapshot(payload));
        setError(null);
      } catch (failure) {
        if (controller.signal.aborted) return;
        setSnapshot(null);
        setError(failure instanceof Error ? failure.message : 'Unable to check semantic search.');
      } finally {
        if (read.current === controller) read.current = null;
      }
    })();
    return () => controller.abort();
  }, [revision]);

  useEffect(() => () => write.current?.abort(), []);

  useEffect(() => {
    if (snapshot?.state !== 'downloading') return;
    const timer = window.setInterval(() => {
      if (!read.current) setRevision((value) => value + 1);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [snapshot?.state]);

  const download = async () => {
    if (write.current || !snapshot || !snapshot.canDownload || snapshot.state !== 'missing') return;
    const controller = new AbortController();
    write.current = controller;
    read.current?.abort();
    setBusy(true);
    try {
      const response = await fetch('/ui/api/embeddings/download', {
        method: 'POST', signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: snapshot.model, confirm: true }),
      });
      const payload: unknown = await response.json();
      if (controller.signal.aborted) return;
      if (!response.ok) throw new Error(extractErrorMessage(payload, 'Model download failed.'));
      setSnapshot(null);
      setRevision((value) => value + 1);
    } catch (failure) {
      if (controller.signal.aborted) return;
      setSnapshot(null);
      setError(failure instanceof Error ? failure.message : 'Model download failed.');
    } finally {
      if (!controller.signal.aborted) {
        write.current = null;
        setBusy(false);
      }
    }
  };

  if (dismissed || (!snapshot && !error)
    || (snapshot && (snapshot.provider !== 'local' || snapshot.state === 'ready'))) return null;

  return (
    <aside aria-label="Semantic search availability" className="shrink-0 border-b border-zinc-700 bg-zinc-900 px-4 py-3 text-sm text-zinc-200">
      <p role="status">
        {error ? `Unable to check or restore semantic search: ${error}`
          : snapshot?.state === 'package_missing' ? 'Semantic search is unavailable: the server is missing the local embeddings library.'
            : snapshot?.state === 'missing' ? `Semantic search is unavailable: model files for ${snapshot.model} are missing.`
              : snapshot?.state === 'downloading' ? 'Semantic search model files are downloading.'
                : 'The local semantic search model could not be loaded.'}
        {' '}Chat remains available.
      </p>
      {snapshot?.state === 'package_missing' ? (
        <p className="mt-1 text-xs text-zinc-400">The server administrator can install the local embeddings dependencies with <code>make install-beta</code> in the SlavikAI server checkout, then restart the server and recheck. Automatic installation is unavailable.</p>
      ) : null}
      {snapshot?.state === 'missing' && !snapshot.canDownload ? (
        <p className="mt-1 text-xs text-zinc-400">Ask the owner to download the model on the SlavikAI server.</p>
      ) : null}
      <div className="mt-2 flex gap-3">
        {snapshot?.state === 'missing' && snapshot.canDownload ? (
          <button type="button" className="rounded-md bg-zinc-100 px-3 py-1.5 text-zinc-950 hover:bg-white disabled:opacity-40" disabled={busy} onClick={() => { void download(); }}>
            {busy ? 'Starting download…' : 'Download model to server'}
          </button>
        ) : null}
        <button type="button" className="rounded-md border border-zinc-600 px-3 py-1.5 hover:bg-zinc-800 disabled:opacity-40" disabled={busy} onClick={() => setRevision((value) => value + 1)}>Recheck semantic search</button>
        <button type="button" className="px-2 py-1.5 text-zinc-400 hover:text-zinc-200" onClick={() => setDismissed(true)}>Dismiss notice</button>
      </div>
    </aside>
  );
}
