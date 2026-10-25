const fields = [
  ['total_chars', 'Total context', 'characters'],
  ['prefs_max_items', 'Preferences', 'items'],
  ['prefs_chars', 'Preferences text', 'characters'],
  ['legacy_notes_chars', 'Legacy notes', 'characters'],
  ['feedback_max_items', 'Feedback', 'items'],
  ['feedback_chars', 'Feedback text', 'characters'],
  ['canonical_memory_chars', 'Saved Memory', 'characters'],
  ['vector_code_chars', 'Retrieved code', 'characters'],
  ['vector_docs_chars', 'Retrieved documents', 'characters'],
  ['workspace_file_chars', 'Workspace files', 'characters'],
  ['pinned_atoms_chars', 'Pinned Memory', 'characters'],
  ['session_summary_chars', 'Session summaries', 'characters'],
] as const;
const isRecord = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);

export function ContextBudgetView({ snapshot }: { snapshot: unknown }) {
  const rawValues = isRecord(snapshot) ? snapshot.values : null;
  const valid = isRecord(snapshot) && snapshot.basis === 'server_configuration_at_refresh'
    && typeof snapshot.source === 'string' && snapshot.source.length > 0
    && isRecord(rawValues) && fields.every(([key]) => typeof rawValues[key] === 'number' && Number.isSafeInteger(rawValues[key]) && (rawValues[key] as number) >= 0);
  if (!valid || !isRecord(rawValues) || !isRecord(snapshot)) return <p className="mt-4 text-xs text-zinc-500">Configured context budget is unavailable.</p>;
  return <section aria-label="Configured context budget" className="mt-5 border-t border-zinc-800 pt-4">
    <h4 className="text-sm font-medium text-zinc-200">Configured context budget · Read only</h4>
    <p className="mt-2 text-xs text-zinc-400">Source: {snapshot.source as string}. Loaded at Settings refresh. Already running sessions may use an earlier snapshot.</p>
    <p className="mt-2 text-xs text-zinc-500">Character and item limits for context assembly, not the model token window or enforced Auto execution budgets. Edit the server configuration to change these values.</p>
    <table className="mt-3 w-full text-left text-xs text-zinc-300">
      <thead><tr><th className="py-2 font-medium">Context part</th><th className="py-2 font-medium">Limit</th><th className="py-2 font-medium">Unit</th></tr></thead>
      <tbody>{fields.map(([key, label, unit]) => <tr key={key} className="border-t border-zinc-800"><th scope="row" className="py-2 font-normal">{label}</th><td>{rawValues[key] as number}</td><td className="text-zinc-500">{unit}</td></tr>)}</tbody>
    </table>
  </section>;
}
