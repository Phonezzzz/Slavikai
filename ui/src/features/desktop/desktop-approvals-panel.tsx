import { useEffect, useRef, useState } from 'react';

type SavedRule = {
  reference: string; effect: 'allow' | 'ask' | 'deny'; source: string;
  tool: string | null; action: string | null; target_kind: 'none' | 'path' | 'hidden';
  target_path: string | null; command_restriction_present: boolean; risk_restriction_present: boolean;
};
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const identifier = (value: unknown) => value === null || (typeof value === 'string' && /^[a-z][a-z0-9_]{0,79}$/.test(value));
const parse = (payload: unknown, subject: string): SavedRule[] | null => {
  if (!record(payload) || payload.preferences_subject !== subject || payload.basis !== 'saved_desktop_rules_at_refresh' || !Array.isArray(payload.rules)) return null;
  const rules: SavedRule[] = [];
  for (const value of payload.rules) {
    if (!record(value) || typeof value.reference !== 'string' || !/^[a-f0-9]{64}$/.test(value.reference)
      || !['allow', 'ask', 'deny'].includes(String(value.effect)) || !['persistent', 'builtin', 'once', 'session'].includes(String(value.source))
      || !identifier(value.tool) || !identifier(value.action) || !['none', 'path', 'hidden'].includes(String(value.target_kind))
      || typeof value.command_restriction_present !== 'boolean' || typeof value.risk_restriction_present !== 'boolean') return null;
    if (value.target_kind === 'path' ? typeof value.target_path !== 'string' || !value.target_path.startsWith('/') : value.target_path !== null) return null;
    rules.push(value as SavedRule);
  }
  return rules;
};

export function DesktopApprovalsPanel({ subject }: { subject: string }) {
  const [rules, setRules] = useState<SavedRule[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const epoch = useRef(0);
  useEffect(() => {
    const generation = ++epoch.current;
    const abort = new AbortController();
    const current = () => epoch.current === generation && !abort.signal.aborted;
    setRules(null); setError(false); setLoading(true);
    void (async () => {
      try {
        if (!/^[a-f0-9]{64}$/.test(subject)) throw new Error();
        const response = await fetch('/ui/api/desktop/approvals?view=settings', { signal: abort.signal });
        const payload: unknown = await response.json();
        if (!current()) return;
        const parsed = parse(payload, subject);
        if (!response.ok || parsed === null) throw new Error();
        setRules(parsed);
      } catch {
        if (current()) setError(true);
      } finally {
        if (current()) setLoading(false);
      }
    })();
    return () => { ++epoch.current; abort.abort(); };
  }, [subject, refresh]);
  return <section aria-label="Saved Desktop rules" className="mt-3 space-y-3 text-xs text-zinc-400">
    <div className="flex items-center justify-between gap-3"><span>Read only</span><button type="button" disabled={loading} onClick={() => setRefresh(value => value + 1)} className="rounded border border-zinc-700 px-3 py-2 disabled:opacity-40">Refresh saved Desktop rules</button></div>
    <p>Source: saved Desktop policy at refresh. This list does not include active once/session approvals or prove that an action will be allowed. DENY rules take precedence.</p>
    <p>Command, risk and non-file target restrictions may be hidden. This view cannot change or revoke access.</p>
    {loading ? <p>Loading saved Desktop rules...</p> : error ? <p role="status">Saved Desktop rules are unavailable.</p> : rules?.length === 0 ? <p>No saved Desktop rules.</p> : rules?.map((rule, index) => <article key={`${rule.reference}:${index}`} className="space-y-2 rounded-lg border border-zinc-800 p-3">
      <p>{rule.tool ?? 'Unavailable tool'}:{rule.action ?? 'Unavailable action'}</p>
      <p>Saved effect: {rule.effect} · Recorded source: {rule.source}</p>
      {rule.target_kind === 'path' ? <p className="break-all">{rule.target_path}</p> : <p>{rule.target_kind === 'none' ? 'Applies to actions without a target.' : 'Target restriction present; contents hidden.'}</p>}
      {rule.command_restriction_present ? <p>Command restriction present; contents hidden.</p> : null}
      {rule.risk_restriction_present ? <p>Risk restriction present; contents hidden.</p> : null}
    </article>)}
  </section>;
}
