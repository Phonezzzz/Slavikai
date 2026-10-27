const fields = [
  ['tts_timeout_seconds', 'TTS HTTP attempt timeout', 'seconds'],
  ['tts_input_code_points', 'TTS input text limit', 'code points'],
  ['tts_request_code_points', 'TTS request text slice', 'code points'],
  ['stt_timeout_seconds', 'Agent STT HTTP attempt timeout', 'seconds'],
] as const;
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);

export function SpeechToolDefaultsView({ snapshot }: { snapshot: unknown }) {
  const values = record(snapshot) ? snapshot.values : null;
  const valid = record(snapshot) && snapshot.basis === 'speech_tool_class_defaults_at_refresh'
    && record(values) && fields.every(([key]) => typeof values[key] === 'number' && Number.isSafeInteger(values[key]) && (values[key] as number) > 0);
  if (!valid || !record(values)) return <p className="mt-4 text-xs text-zinc-500">Speech tool defaults are unavailable.</p>;
  return <section aria-label="Speech tool defaults" className="mt-4 rounded-xl border border-zinc-800 p-4">
    <h4 className="text-sm font-medium text-zinc-200">Speech tool defaults · Read only</h4>
    <p className="mt-2 text-xs text-zinc-400">Source: TTS and STT tool configuration class defaults, read at Settings refresh. Existing tool instances may use different numeric limits.</p>
    <table className="mt-3 w-full text-left text-xs text-zinc-300">
      <thead><tr><th>Default</th><th>Value</th><th>Unit</th></tr></thead>
      <tbody>{fields.map(([key, label, unit]) => <tr key={key} className="border-t border-zinc-800"><th scope="row" className="py-2 font-normal">{label}</th><td>{values[key] as number}</td><td>{unit}</td></tr>)}</tbody>
    </table>
    <p className="mt-3 text-xs text-zinc-500">TTS text is measured in Unicode code points after trimming. Long text can create several HTTP requests; an attempt timeout is not a total audio deadline.</p>
    <p className="mt-2 text-xs text-zinc-500">Browser dictation uses a separate endpoint; the Agent STT timeout shown here is not its setting. Environment-selected model, voice and format are shown in Speech backend; request overrides are separate. These defaults do not report credentials, capability or successful speech inference.</p>
  </section>;
}
