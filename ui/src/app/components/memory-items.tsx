import { useEffect, useRef, useState } from 'react';

type Item = {
  atom_id: string; revision: string; stable_key: string; claim_type: string;
  summary_text: string; value_json: unknown; status: string; pinned: boolean;
};
type Preview = { item: Item; operation_id: string; expires_at: number };
export type MemoryWrites = Map<string, Set<Promise<unknown>>>;
const button = 'rounded-lg border border-zinc-700 px-3 py-2 text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-50';
const record = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const isItem = (value: unknown): value is Item => record(value)
  && ['atom_id', 'revision', 'stable_key', 'claim_type', 'summary_text', 'status'].every(key => typeof value[key] === 'string')
  && typeof value.pinned === 'boolean' && 'value_json' in value;

async function payload(response: Response): Promise<Record<string, unknown>> {
  if (!response.ok) {
    if (response.status === 409) throw new Error('Memory changed. Reload and review the item again.');
    if (response.status === 410) throw new Error('Deletion preview expired. Review the item again.');
    if (response.status === 404) throw new Error('Memory item or deletion preview is no longer available.');
    throw new Error('Memory request failed. Reload to check the current state.');
  }
  const data: unknown = await response.json();
  if (!record(data)) throw new Error('Memory response is unavailable.');
  return data;
}

export function MemoryItems({ subject, pendingWrites }: { subject: string; pendingWrites: MemoryWrites }) {
  const [items, setItems] = useState<Item[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const identity = useRef<string | null>(null);
  const current = useRef(true);
  const generation = useRef(0);
  const request = useRef<AbortController | null>(null);
  const lock = useRef(false);

  const check = (data: Record<string, unknown>, first = false) => {
    if (data.preferences_subject !== subject || typeof data.subject !== 'string' || !data.subject
      || (!first && data.subject !== identity.current)) {
      setItems([]); setCursor(null); setPreview(null); identity.current = null;
      throw new Error('Account changed. Refresh Settings before viewing Memory.');
    }
  };
  const load = async (after: string | null = null) => {
    if (lock.current) return;
    const epoch = generation.current;
    const valid = () => current.current && epoch === generation.current;
    lock.current = true;
    setLoading(true); setError(null); setPreview(null);
    const controller = new AbortController();
    request.current = controller;
    try {
      // A reopened panel must read after an earlier confirm finishes for this account.
      await Promise.allSettled([...(pendingWrites.get(subject) ?? [])]);
      if (!valid()) return;
      const data = await payload(await fetch(`/ui/api/memory/items?limit=50${after ? `&cursor=${encodeURIComponent(after)}` : ''}`, { signal: controller.signal, cache: 'no-store' }));
      if (!valid()) return;
      check(data, after === null);
      if (!Array.isArray(data.items) || !data.items.every(isItem) || !(data.next_cursor === null || typeof data.next_cursor === 'string')) throw new Error('Memory list is unavailable.');
      identity.current = data.subject as string;
      const next = data.items as Item[];
      setItems(old => after ? [...old.filter(value => !next.some(item => item.atom_id === value.atom_id)), ...next] : next);
      setCursor(data.next_cursor as string | null);
    } catch (reason) {
      if (valid()) setError(reason instanceof Error ? reason.message : 'Memory list is unavailable.');
    } finally {
      if (valid()) { lock.current = false; setLoading(false); }
    }
  };
  useEffect(() => {
    current.current = true;
    void load();
    return () => { current.current = false; generation.current += 1; lock.current = false; request.current?.abort(); };
  }, []);

  const review = async (item: Item) => {
    if (lock.current) return;
    const epoch = generation.current;
    const valid = () => current.current && epoch === generation.current;
    lock.current = true;
    setBusy(true); setError(null); setNotice(null); setPreview(null);
    const controller = new AbortController(); request.current = controller;
    try {
      const data = await payload(await fetch('/ui/api/memory/delete/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ atom_id: item.atom_id, revision: item.revision }), signal: controller.signal }));
      if (!valid()) return;
      check(data);
      if (!isItem(data.item) || data.item.atom_id !== item.atom_id || data.item.revision !== item.revision
        || data.scope !== 'canonical_atom_and_atom_vector' || typeof data.operation_id !== 'string' || !data.operation_id
        || typeof data.expires_at !== 'number' || data.expires_at * 1000 <= Date.now()) throw new Error('Deletion preview is unavailable or expired. Reload and review again.');
      setPreview({ item: data.item, operation_id: data.operation_id, expires_at: data.expires_at });
    } catch (reason) {
      if (valid()) setError(reason instanceof Error ? reason.message : 'Deletion preview is unavailable.');
    } finally {
      if (valid()) { lock.current = false; setBusy(false); }
    }
  };
  const confirm = async () => {
    if (lock.current || !preview) return;
    if (preview.expires_at * 1000 <= Date.now()) { setPreview(null); setError('Deletion preview expired. Review the item again.'); return; }
    const epoch = generation.current;
    const valid = () => current.current && epoch === generation.current;
    lock.current = true; setBusy(true); setError(null);
    const writes = pendingWrites.get(subject) ?? new Set<Promise<unknown>>();
    pendingWrites.set(subject, writes);
    // Do not abort an explicit write when the panel closes: its completion fences the next read.
    const operation = fetch('/ui/api/memory/delete/confirm', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ operation_id: preview.operation_id, confirm: true }) }).then(payload);
    writes.add(operation);
    let deleted = false;
    try {
      const data = await operation;
      if (!valid()) return;
      check(data);
      if (data.status !== 'deleted') throw new Error('Deletion result is unavailable. Reload to check Memory.');
      setNotice('Memory item deleted.'); setPreview(null); deleted = true;
    } catch (reason) {
      if (valid()) { setPreview(null); setError(`${reason instanceof Error ? reason.message : 'Deletion result is unavailable.'} A failed connection does not prove that deletion was cancelled.`); }
    } finally {
      writes.delete(operation);
      if (!writes.size) pendingWrites.delete(subject);
      if (valid()) { lock.current = false; setBusy(false); }
    }
    if (deleted && valid()) await load();
  };
  return (
    <section aria-label="Your Memory" className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-medium text-zinc-100">Your Memory</h3>
        <button type="button" className={button} disabled={loading || busy} onClick={() => { setNotice(null); void load(); }}>Reload Memory</button>
      </div>
      <p className="mt-2 text-xs text-zinc-400">Saved Memory for your account, including retired items. No model or active chat is required.</p>
      <p className="mt-2 text-xs text-zinc-500">Delete removes only the selected saved item and its Memory search vector. Chat history, inbox, preferences, traces and backups remain. Existing chat context may still contain a copy. This is not secure disk erasure.</p>
      {loading ? <p role="status" className="mt-3 text-xs text-zinc-400">Loading Memory…</p> : null}
      {busy && !preview ? <p role="status" className="mt-3 text-xs text-zinc-400">Preparing deletion preview…</p> : null}
      {error ? <p role="alert" className="mt-3 text-xs text-red-300">{error}</p> : null}
      {notice ? <p role="status" className="mt-3 text-xs text-emerald-300">{notice}</p> : null}
      {!loading && !error && !items.length ? <p className="mt-3 text-xs text-zinc-400">No saved Memory items.</p> : null}
      <div className="mt-3 space-y-3">
        {items.map(item => <article key={item.atom_id} className="rounded-lg border border-zinc-800 p-3">
          <p className="whitespace-pre-wrap break-words text-sm text-zinc-200">{item.summary_text || item.stable_key}</p>
          <p className="mt-1 text-xs text-zinc-500">{item.claim_type} · {item.status}{item.pinned ? ' · Pinned' : ''}</p>
          <pre className="my-2 whitespace-pre-wrap break-all text-xs text-zinc-400">{JSON.stringify(item.value_json, null, 2)}</pre>
          <button type="button" className={button} disabled={loading || busy || !!preview} onClick={() => void review(item)}>Review deletion</button>
          {preview?.item.atom_id === item.atom_id ? <div role="group" aria-label="Deletion preview" className="mt-4 rounded-lg border border-red-900/60 p-3">
            <h4 className="text-sm text-zinc-100">Delete this Memory item?</h4>
            <p className="mt-2 whitespace-pre-wrap break-words text-sm text-zinc-200">{preview.item.summary_text || preview.item.stable_key}</p>
            <pre className="my-2 whitespace-pre-wrap break-all text-xs text-zinc-400">{JSON.stringify(preview.item.value_json, null, 2)}</pre>
            <p className="mb-3 text-xs text-zinc-400">This permanently removes this saved item and its Memory search vector. The retained data listed above is unchanged. Confirmation expires after five minutes.</p>
            <div className="flex gap-2">
              <button type="button" className={button} disabled={busy} onClick={() => void confirm()}>{busy ? 'Deleting…' : 'Confirm delete'}</button>
              <button type="button" className={button} disabled={busy} onClick={() => setPreview(null)}>Cancel deletion</button>
            </div>
          </div> : null}
        </article>)}
      </div>
      {cursor ? <button type="button" className={`${button} mt-3`} disabled={loading || busy || !!preview} onClick={() => void load(cursor)}>Load more Memory</button> : null}
    </section>
  );
}
