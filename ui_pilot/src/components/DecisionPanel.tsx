import type { DecisionPacketView } from "../types";

type DecisionPanelProps = {
  decision: DecisionPacketView | null;
};

export default function DecisionPanel({ decision }: DecisionPanelProps) {
  if (!decision) {
    return (
      <section className="flex flex-col gap-4 rounded-3xl border border-slate-800/80 bg-slate-900/60 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-100">Decision</h2>
          <span className="text-xs text-slate-500">none</span>
        </div>
        <div className="rounded-2xl border border-dashed border-slate-800/70 bg-slate-900/60 px-4 py-6 text-sm text-slate-400">
          Нет активного DecisionPacket.
        </div>
      </section>
    );
  }

  const defaultId = decision.default_option_id ?? null;

  return (
    <section className="flex flex-col gap-4 rounded-3xl border border-slate-800/80 bg-slate-900/60 p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Decision</p>
          <h2 className="text-lg font-semibold text-slate-100">Packet</h2>
        </div>
        <span className="text-xs text-slate-500">{decision.id}</span>
      </div>

      <div className="rounded-2xl border border-slate-800/80 bg-slate-950/40 px-4 py-3">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-500">Summary</div>
        <div className="mt-2 text-sm text-slate-100">{decision.summary}</div>
      </div>

      <div className="rounded-2xl border border-slate-800/80 bg-slate-950/40 px-4 py-3">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-500">Reason</div>
        <div className="mt-2 text-sm text-slate-100">{decision.reason}</div>
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between text-xs uppercase tracking-[0.3em] text-slate-500">
          <span>Options</span>
          {defaultId ? <span className="text-[10px]">default: {defaultId}</span> : null}
        </div>
        <div className="flex flex-col gap-3">
          {decision.options.map((option) => {
            const isDefault = defaultId === option.id;
            return (
              <div
                key={option.id}
                className={`rounded-2xl border px-3 py-3 ${
                  isDefault
                    ? "border-emerald-400/60 bg-emerald-950/30"
                    : "border-slate-800/80 bg-slate-950/40"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-slate-100">{option.title}</div>
                    <div className="mt-1 text-xs text-slate-400">{option.action}</div>
                  </div>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] uppercase tracking-wide ${
                      option.risk === "high"
                        ? "bg-rose-500/20 text-rose-300"
                        : option.risk === "medium"
                          ? "bg-amber-500/20 text-amber-300"
                          : "bg-emerald-500/20 text-emerald-300"
                    }`}
                  >
                    {option.risk}
                  </span>
                </div>
                {isDefault ? (
                  <div className="mt-2 text-[11px] uppercase tracking-[0.3em] text-emerald-200">
                    default
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
