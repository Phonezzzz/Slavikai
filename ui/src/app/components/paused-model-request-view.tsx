const options = [
  ['temperature', 'Temperature', 'number'], ['top_p', 'Top P', 'number'],
  ['max_tokens', 'Maximum output tokens', 'number'], ['mode', 'Configured mode', 'mode'],
  ['thinking_enabled', 'Thinking requested', 'bool'], ['reasoning_effort', 'Reasoning effort', 'effort'],
  ['reasoning_summary', 'Reasoning summary requested', 'bool'],
  ['reasoning_summary_wait', 'Wait for reasoning summary', 'bool'], ['diffusing', 'Diffusing requested', 'bool'],
  ['web_search_enabled', 'Web search requested', 'bool'],
] as const;
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
function display(value: unknown, kind: string): string {
  if (value === null) return 'Not set';
  if (kind === 'number' && typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= Number.MAX_SAFE_INTEGER) return String(value);
  if (kind === 'bool' && typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (kind === 'mode' && (value === 'default' || value === 'planner')) return value;
  if (kind === 'effort' && typeof value === 'string' && ['instant', 'low', 'medium', 'high'].includes(value)) return value;
  return 'Unavailable';
}
export function PausedModelRequestView({ snapshot, sessionId }: { snapshot: unknown; sessionId: string | null }) {
  if (!record(snapshot) || snapshot.session_id !== sessionId || snapshot.basis !== 'paused_approval_model_request_at_refresh' || snapshot.state !== 'available' || typeof snapshot.decision_id !== 'string' || !snapshot.decision_id || !record(snapshot.values) || !record(snapshot.identity) || typeof snapshot.identity.provider !== 'string' || typeof snapshot.identity.model !== 'string' || !snapshot.identity.provider.trim() || !snapshot.identity.model.trim() || !Array.isArray(snapshot.unavailable_fields)) return <p className="mt-2 text-xs text-zinc-500">Paused model request is unavailable.</p>;
  const values = snapshot.values;
  const unavailable = snapshot.unavailable_fields;
  return <section aria-label="Paused model request" className="mt-3 border-t border-zinc-800 pt-3">
    <h4 className="text-sm font-medium text-zinc-200">Paused model request · Read only</h4>
    <p className="mt-2 text-xs text-zinc-400">{snapshot.identity.provider} / {snapshot.identity.model}</p>
    <p className="mt-2 text-xs text-zinc-500">Source: captured approval continuation in this session, read at refresh. These options describe that paused request, not saved defaults or a currently executing request.</p>
    <table className="mt-3 w-full text-left text-xs text-zinc-300"><thead><tr><th>Option</th><th>Captured value</th></tr></thead><tbody>{options.map(([key, label, kind]) => <tr key={key} className="border-t border-zinc-800"><th scope="row" className="py-2 font-normal">{label}</th><td>{unavailable.includes(key) ? 'Unavailable' : display(values[key], kind)}</td></tr>)}</tbody></table>
    <p className="mt-3 text-xs text-zinc-500">This is not a provider acceptance, search execution, permission or resumability check. Approval settlement, cancellation or restart can remove the checkpoint; refresh to read again. Prompt, history, credentials, endpoints and tool arguments are not displayed.</p>
  </section>;
}
