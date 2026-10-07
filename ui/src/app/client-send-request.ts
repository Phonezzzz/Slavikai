export class PendingSendConflict extends Error {
  constructor(public readonly key: string) { super("Предыдущий запрос ещё не подтверждён. Для новой отправки сначала отбросьте его явно."); }
}

type PendingSend = {
  key: string;
  fingerprint: string;
  regenerated: boolean;
};

const storageKey = (sessionId: string): string => `slavikai:pending-send:${sessionId}`;

function readPending(sessionId: string): PendingSend | null {
  const value = localStorage.getItem(storageKey(sessionId));
  if (value === null) return null;
  const record: unknown = JSON.parse(value);
  if (!record || typeof record !== 'object') throw new Error('Invalid pending send identity.');
  const parsed = record as Partial<PendingSend>;
  if (typeof parsed.key !== 'string' || typeof parsed.fingerprint !== 'string' ||
      typeof parsed.regenerated !== 'boolean') throw new Error('Invalid pending send identity.');
  return parsed as PendingSend;
}

export async function prepareClientSend(sessionId: string, body: string): Promise<PendingSend> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(body));
  const fingerprint = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
  const existing = readPending(sessionId);
  if (existing?.fingerprint === fingerprint) return existing;
  if (existing) throw new PendingSendConflict(existing.key);
  const pending = { key: crypto.randomUUID(), fingerprint, regenerated: false };
  // Persist only identity/hash before dispatch; no message or attachment bytes.
  localStorage.setItem(storageKey(sessionId), JSON.stringify(pending));
  return pending;
}

export function markClientSendRegenerated(sessionId: string, key: string): void {
  const pending = readPending(sessionId);
  if (pending?.key === key) {
    localStorage.setItem(storageKey(sessionId), JSON.stringify({ ...pending, regenerated: true }));
  }
}

export function acknowledgeClientSend(sessionId: string, key: string): void {
  if (readPending(sessionId)?.key === key) localStorage.removeItem(storageKey(sessionId));
}
