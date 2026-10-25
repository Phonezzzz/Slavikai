const options = [
  ['temperature', 'Temperature', 'number'], ['top_p', 'Top P', 'number'],
  ['max_tokens', 'Maximum output tokens', 'number'], ['mode', 'Configured mode', 'mode'],
  ['thinking_enabled', 'Thinking requested', 'bool'], ['reasoning_effort', 'Reasoning effort', 'effort'],
  ['reasoning_summary', 'Reasoning summary requested', 'bool'],
  ['reasoning_summary_wait', 'Wait for reasoning summary', 'bool'], ['diffusing', 'Diffusing requested', 'bool'],
] as const;
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const display = (value: unknown, kind: string): string => {
  if (value === null) return 'Not set';
  if (kind === 'number' && typeof value === 'number' && Number.isFinite(value)) return String(value);
  if (kind === 'bool' && typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (kind === 'mode' && (value === 'default' || value === 'planner')) return value;
  if (kind === 'effort' && ['instant', 'low', 'medium', 'high'].includes(value as string)) return value as string;
  return 'Unavailable';
};
export function ModelConfigView({ snapshot }: { snapshot: unknown }) {
  if (!record(snapshot) || snapshot.state !== 'available' || snapshot.basis !== 'loaded_default_model_at_refresh'
    || !record(snapshot.values) || !record(snapshot.presence) || !record(snapshot.identity)
    || typeof snapshot.source !== 'string' || typeof snapshot.identity.provider !== 'string' || typeof snapshot.identity.model !== 'string') return <p className="mt-4 text-xs text-zinc-500">Saved model options are unavailable.</p>;
  const values = snapshot.values, presence = snapshot.presence;
  const unavailable = Array.isArray(snapshot.unavailable_fields) ? snapshot.unavailable_fields : [];
  return <section aria-label="Saved model options" className="mt-4 border-t border-zinc-800 pt-4">
    <h4 className="text-sm font-medium text-zinc-200">Saved model options · Read only</h4>
    <p className="mt-2 text-xs text-zinc-400">{snapshot.identity.provider} / {snapshot.identity.model}</p>
    <p className="mt-2 text-xs text-zinc-500">Source: {snapshot.source}. Loaded at Settings refresh, separately from your selection draft. Already running chats may use different options. Provider support varies; this is not a capability check.</p>
    <table className="mt-3 w-full text-left text-xs text-zinc-300"><thead><tr><th className="py-2">Option</th><th>Loaded value</th></tr></thead><tbody>
      {options.map(([key, label, kind]) => <tr key={key} className="border-t border-zinc-800"><th scope="row" className="py-2 font-normal">{label}</th><td>{unavailable.includes(key) ? 'Unavailable' : display(values[key], kind)}</td></tr>)}
      {([['api_key', 'Credential override present'], ['base_url', 'Endpoint override present'], ['extra_headers', 'Additional headers present'], ['system_prompt', 'Prompt override present']] as const).map(([key, label]) => <tr key={key} className="border-t border-zinc-800"><th scope="row" className="py-2 font-normal">{label}</th><td>{typeof presence[key] === 'boolean' ? presence[key] ? 'Yes' : 'No' : 'Unavailable'}</td></tr>)}
    </tbody></table>
    <p className="mt-3 text-xs text-zinc-500">Presence flags describe the saved configuration, not active credential precedence. Credential, endpoint, header and prompt contents are not displayed. Change these options in the server configuration.</p>
    <p className="mt-2 text-xs text-zinc-500">Web search option: the configuration loader does not restore this field. No active value is claimed.</p>
  </section>;
}
