const fields = [
  ['top_k', 'Candidates considered', 'candidates'],
  ['timeout_seconds', 'HTTP attempt timeout', 'seconds'],
  ['response_content_bytes', 'Response content read budget', 'bytes'],
] as const;
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);

export function SearchBackendDefaultsView({ snapshot }: { snapshot: unknown }) {
  const values = record(snapshot) ? snapshot.values : null;
  const valid = record(snapshot) && snapshot.basis === 'backend_class_defaults_at_refresh'
    && record(values) && fields.every(([key]) => typeof values[key] === 'number' && Number.isSafeInteger(values[key]) && (values[key] as number) > 0);
  if (!valid || !record(values)) return <p className="mt-4 text-xs text-zinc-500">Search backend defaults are unavailable.</p>;
  return <section aria-label="Search backend defaults" className="mt-4 rounded-xl border border-zinc-800 p-4">
    <h4 className="text-sm font-medium text-zinc-200">Search backend defaults · Read only</h4>
    <p className="mt-2 text-xs text-zinc-400">Source: search backend configuration class defaults, read at Settings refresh. Existing tool instances may use different values.</p>
    <table className="mt-3 w-full text-left text-xs text-zinc-300">
      <thead><tr><th>Default</th><th>Value</th><th>Unit</th></tr></thead>
      <tbody>{fields.map(([key, label, unit]) => <tr key={key} className="border-t border-zinc-800"><th scope="row" className="py-2 font-normal">{label}</th><td>{values[key] as number}</td><td>{unit}</td></tr>)}</tbody>
    </table>
    <p className="mt-3 text-xs text-zinc-500">Valid results may be fewer than the candidate limit. Search attempts can retry, so the timeout is not a total deadline. Truncation markers are added beyond the content read budget; it is not a strict output-size limit.</p>
    <p className="mt-2 text-xs text-zinc-500">These defaults are separate from model-native web search and its saved request flag. This view does not report active chat options, credentials, search readiness, capability or an executed search.</p>
  </section>;
}
