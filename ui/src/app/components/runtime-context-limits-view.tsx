const fields = [
  ['short_term_max_messages', 'Transient short-term buffer', 'user/assistant entries'],
  ['summary_max_messages', 'Summary input, latest nonempty messages', 'user/assistant entries'],
  ['summary_message_code_points', 'Summary input, content per message', 'code points'],
] as const;
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);

export function RuntimeContextLimitsView({ snapshot }: { snapshot: unknown }) {
  const values = record(snapshot) ? snapshot.values : null;
  const valid = record(snapshot) && snapshot.basis === 'runtime_context_constants_at_refresh'
    && record(values) && fields.every(([key]) => typeof values[key] === 'number' && Number.isSafeInteger(values[key]) && (values[key] as number) > 0);
  if (!valid || !record(values)) return <p className="mt-4 text-xs text-zinc-500">Runtime context limits are unavailable.</p>;
  return <section aria-label="Runtime context limits" className="mt-5 border-t border-zinc-800 pt-4">
    <h4 className="text-sm font-medium text-zinc-200">Runtime context limits · Read only</h4>
    <p className="mt-2 text-xs text-zinc-400">Source: server short-term buffer and session summarizer constants, read at Settings refresh.</p>
    <table className="mt-3 w-full text-left text-xs text-zinc-300">
      <thead><tr><th>Limit</th><th>Value</th><th>Unit</th></tr></thead>
      <tbody>{fields.map(([key, label, unit]) => <tr key={key} className="border-t border-zinc-800"><th scope="row" className="py-2 font-normal">{label}</th><td>{values[key] as number}</td><td>{unit}</td></tr>)}</tbody>
    </table>
    <p className="mt-3 text-xs text-zinc-500">The transient buffer limit does not limit saved chat history or every model request. Browser session-summary preview reads saved chat history separately. Saving that preview to Memory requires another confirmation.</p>
    <p className="mt-2 text-xs text-zinc-500">Summary content is measured in Unicode code points, including serialized attachment text; role prefixes are additional. These are input slices, not retention periods, the model token window, remaining capacity or a guarantee of a complete summary.</p>
  </section>;
}
