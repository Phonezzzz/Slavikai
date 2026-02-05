import type { UIEvent } from "../types";

type DebugPanelProps = {
  events: UIEvent[];
  traceId: string | null;
  traceEvents: unknown[];
  toolCalls: unknown[];
  loading: boolean;
};

const toPreview = (value: unknown): string => {
  try {
    const raw = JSON.stringify(value ?? {}, null, 0);
    return raw.length > 180 ? `${raw.slice(0, 180)}...` : raw;
  } catch {
    return "{unserializable}";
  }
};

export default function DebugPanel({
  events,
  traceId,
  traceEvents,
  toolCalls,
  loading,
}: DebugPanelProps) {
  return (
    <div className="flex flex-col gap-4">
      <section className="flex flex-col gap-4 rounded-3xl border border-neutral-800/80 bg-neutral-900/60 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-neutral-100">Event log</h2>
          <span className="text-xs text-neutral-500">{events.length} items</span>
        </div>
        <div className="flex min-h-[24vh] flex-col gap-3 overflow-y-auto rounded-3xl border border-neutral-800/80 bg-neutral-950/40 p-3 font-mono text-xs">
          {events.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-neutral-800/70 bg-neutral-900/60 px-4 py-6 text-neutral-400">
              Awaiting SSE events.
            </div>
          ) : (
            events.map((evt) => (
              <div
                key={evt.id}
                className="rounded-2xl border border-neutral-800/80 bg-neutral-900/80 px-3 py-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-neutral-200">{evt.type}</span>
                  <span className="text-[10px] text-neutral-500">{evt.ts}</span>
                </div>
                <div className="mt-1 text-neutral-400">{toPreview(evt.payload)}</div>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="flex flex-col gap-4 rounded-3xl border border-neutral-800/80 bg-neutral-900/60 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-neutral-100">Trace</h2>
          <span className="text-xs text-neutral-500">{traceId ?? "n/a"}</span>
        </div>
        <div className="rounded-2xl border border-neutral-800/80 bg-neutral-950/40 px-3 py-2 text-xs text-neutral-300">
          {loading ? "Loading trace..." : `Events: ${traceEvents.length}`}
        </div>
        <div className="max-h-40 space-y-2 overflow-y-auto rounded-2xl border border-neutral-800/80 bg-neutral-950/40 p-2 font-mono text-[11px]">
          {traceEvents.length === 0 ? (
            <div className="text-neutral-500">No trace events.</div>
          ) : (
            traceEvents.slice(-20).map((item, index) => (
              <div key={`${traceId ?? "trace"}-${index}`} className="text-neutral-400">
                {toPreview(item)}
              </div>
            ))
          )}
        </div>
      </section>

      <section className="flex flex-col gap-4 rounded-3xl border border-neutral-800/80 bg-neutral-900/60 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-neutral-100">Tool calls</h2>
          <span className="text-xs text-neutral-500">{toolCalls.length}</span>
        </div>
        <div className="max-h-44 space-y-2 overflow-y-auto rounded-2xl border border-neutral-800/80 bg-neutral-950/40 p-2 font-mono text-[11px]">
          {toolCalls.length === 0 ? (
            <div className="text-neutral-500">No tool calls.</div>
          ) : (
            toolCalls.slice(-20).map((item, index) => (
              <div key={`tool-${index}`} className="text-neutral-400">
                {toPreview(item)}
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}
