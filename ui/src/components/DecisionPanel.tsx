import type { DecisionPacketView } from "../types";

type ProjectCommand = "find" | "index";

type DecisionPanelProps = {
  decision: DecisionPacketView | null;
  projectCommand: ProjectCommand;
  projectArgs: string;
  projectBusy: boolean;
  onProjectCommandChange: (value: ProjectCommand) => void;
  onProjectArgsChange: (value: string) => void;
  onProjectRun: () => void;
};

export default function DecisionPanel({
  decision,
  projectCommand,
  projectArgs,
  projectBusy,
  onProjectCommandChange,
  onProjectArgsChange,
  onProjectRun,
}: DecisionPanelProps) {
  const projectHint =
    projectCommand === "find"
      ? "Поиск по индексу (пример: payment timeout)"
      : "Путь внутри sandbox/project (пусто = весь workspace)";
  if (!decision) {
    return (
      <section className="flex flex-col gap-4 rounded-3xl border border-neutral-800/80 bg-neutral-900/60 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-neutral-100">Decision</h2>
          <span className="text-xs text-neutral-500">none</span>
        </div>
        <div className="rounded-2xl border border-neutral-800/80 bg-neutral-950/40 px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="text-xs uppercase tracking-[0.3em] text-neutral-500">Tool</div>
            <span className="rounded-full bg-neutral-800/80 px-2 py-0.5 text-[10px] uppercase tracking-wide text-neutral-300">
              project
            </span>
          </div>
          <div className="mt-3 flex items-center gap-2">
            <select
              value={projectCommand}
              onChange={(event) => onProjectCommandChange(event.target.value as ProjectCommand)}
              className="rounded-full border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200"
            >
              <option value="find">find</option>
              <option value="index">index</option>
            </select>
            <button
              type="button"
              onClick={onProjectRun}
              disabled={projectBusy}
              className="rounded-full border border-neutral-700 bg-neutral-900 px-3 py-1 text-xs text-neutral-200 disabled:opacity-50"
            >
              {projectBusy ? "..." : "Run"}
            </button>
          </div>
          <input
            value={projectArgs}
            onChange={(event) => onProjectArgsChange(event.target.value)}
            placeholder={projectHint}
            className="mt-2 w-full rounded-xl border border-neutral-800 bg-neutral-900 px-3 py-2 text-xs text-neutral-200 placeholder:text-neutral-500"
          />
        </div>
        <div className="rounded-2xl border border-dashed border-neutral-800/70 bg-neutral-900/60 px-4 py-6 text-sm text-neutral-400">
          Нет активного DecisionPacket.
        </div>
      </section>
    );
  }

  const defaultId = decision.default_option_id ?? null;

  return (
    <section className="flex flex-col gap-4 rounded-3xl border border-neutral-800/80 bg-neutral-900/60 p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-neutral-500">Decision</p>
          <h2 className="text-lg font-semibold text-neutral-100">Packet</h2>
        </div>
        <span className="text-xs text-neutral-500">{decision.id}</span>
      </div>

      <div className="rounded-2xl border border-neutral-800/80 bg-neutral-950/40 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="text-xs uppercase tracking-[0.3em] text-neutral-500">Tool</div>
          <span className="rounded-full bg-neutral-800/80 px-2 py-0.5 text-[10px] uppercase tracking-wide text-neutral-300">
            project
          </span>
        </div>
        <div className="mt-3 flex items-center gap-2">
          <select
            value={projectCommand}
            onChange={(event) => onProjectCommandChange(event.target.value as ProjectCommand)}
            className="rounded-full border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200"
          >
            <option value="find">find</option>
            <option value="index">index</option>
          </select>
          <button
            type="button"
            onClick={onProjectRun}
            disabled={projectBusy}
            className="rounded-full border border-neutral-700 bg-neutral-900 px-3 py-1 text-xs text-neutral-200 disabled:opacity-50"
          >
            {projectBusy ? "..." : "Run"}
          </button>
        </div>
        <input
          value={projectArgs}
          onChange={(event) => onProjectArgsChange(event.target.value)}
          placeholder={projectHint}
          className="mt-2 w-full rounded-xl border border-neutral-800 bg-neutral-900 px-3 py-2 text-xs text-neutral-200 placeholder:text-neutral-500"
        />
      </div>

      <div className="rounded-2xl border border-neutral-800/80 bg-neutral-950/40 px-4 py-3">
        <div className="text-xs uppercase tracking-[0.3em] text-neutral-500">Summary</div>
        <div className="mt-2 text-sm text-neutral-100">{decision.summary}</div>
      </div>

      <div className="rounded-2xl border border-neutral-800/80 bg-neutral-950/40 px-4 py-3">
        <div className="text-xs uppercase tracking-[0.3em] text-neutral-500">Reason</div>
        <div className="mt-2 text-sm text-neutral-100">{decision.reason}</div>
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between text-xs uppercase tracking-[0.3em] text-neutral-500">
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
                    : "border-neutral-800/80 bg-neutral-950/40"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-neutral-100">{option.title}</div>
                    <div className="mt-1 text-xs text-neutral-400">{option.action}</div>
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
