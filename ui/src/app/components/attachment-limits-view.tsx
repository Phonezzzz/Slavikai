const fields = [
  ['max_items', 'Attachments per message', 'items'],
  ['max_item_code_points', 'Content per attachment', 'code points'],
  ['max_total_code_points', 'Total attachment content', 'code points'],
] as const;
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);

export function AttachmentLimitsView({ snapshot }: { snapshot: unknown }) {
  const values = record(snapshot) ? snapshot.values : null;
  const valid = record(snapshot) && snapshot.basis === 'server_request_validation_at_refresh'
    && record(values) && fields.every(([key]) => typeof values[key] === 'number' && Number.isSafeInteger(values[key]) && (values[key] as number) > 0);
  if (!valid || !record(values)) return <p className="mt-4 text-xs text-zinc-500">Server attachment limits are unavailable.</p>;
  return <section aria-label="Server attachment limits" className="mt-4 rounded-xl border border-zinc-800 p-4">
    <h4 className="text-sm font-medium text-zinc-200">Server attachment limits · Read only</h4>
    <p className="mt-2 text-xs text-zinc-400">Source: server attachment validator, read at Settings refresh. These limits apply to each submitted chat message.</p>
    <table className="mt-3 w-full text-left text-xs text-zinc-300">
      <thead><tr><th>Limit</th><th>Value</th><th>Unit</th></tr></thead>
      <tbody>{fields.map(([key, label, unit]) => <tr key={key} className="border-t border-zinc-800"><th scope="row" className="py-2 font-normal">{label}</th><td>{values[key] as number}</td><td>{unit}</td></tr>)}</tbody>
    </table>
    <p className="mt-3 text-xs text-zinc-500">Unicode code points count content text, including an encoded image data URL. They are not file bytes or model tokens. Browser preparation limits may differ. These values do not establish image support or provider capability.</p>
  </section>;
}
