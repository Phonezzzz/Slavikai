const fields = [
  ['timeout_seconds', 'Subprocess timeout argument', 'seconds'],
  ['output_code_points', 'Post-capture output truncation threshold', 'code points'],
] as const;
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);

export function ShellNumericDefaultsView({ snapshot }: { snapshot: unknown }) {
  const values = record(snapshot) ? snapshot.values : null;
  const valid = record(snapshot) && snapshot.basis === 'shell_class_defaults_at_refresh'
    && record(values) && fields.every(([key]) => typeof values[key] === 'number' && Number.isSafeInteger(values[key]) && (values[key] as number) > 0);
  if (!valid || !record(values)) return <p className="mt-4 text-xs text-zinc-500">Shell numeric defaults are unavailable.</p>;
  return <section aria-label="Shell numeric defaults" className="mt-4 rounded-xl border border-zinc-800 p-4">
    <h4 className="text-sm font-medium text-zinc-200">Shell numeric defaults · Read only</h4>
    <p className="mt-2 text-xs text-zinc-400">Source: ShellConfig class defaults, read at Settings refresh. The existing JSON configuration can override these values for each subprocess operation.</p>
    <table className="mt-3 w-full text-left text-xs text-zinc-300">
      <thead><tr><th>Default</th><th>Value</th><th>Unit</th></tr></thead>
      <tbody>{fields.map(([key, label, unit]) => <tr key={key} className="border-t border-zinc-800"><th scope="row" className="py-2 font-normal">{label}</th><td>{values[key] as number}</td><td>{unit}</td></tr>)}</tbody>
    </table>
    <p className="mt-3 text-xs text-zinc-500">ShellTool combines stdout and stderr before truncation; TerminalTool oneshot truncates each stream separately. The threshold counts Unicode code points after capture; added truncation markers can exceed it. This is not a capture memory cap or a strict returned-output limit.</p>
    <p className="mt-2 text-xs text-zinc-500">These defaults do not describe PTY or Desktop sessions, a total task deadline, effective grants, command allowlist or working root.</p>
  </section>;
}
