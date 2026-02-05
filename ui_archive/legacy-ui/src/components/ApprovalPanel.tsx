import type { ApprovalRequestView } from "../types";

type ApprovalPanelProps = {
  approval: ApprovalRequestView | null;
  approving: boolean;
  onApproveRetry: () => void;
  onDismiss: () => void;
};

export default function ApprovalPanel({
  approval,
  approving,
  onApproveRetry,
  onDismiss,
}: ApprovalPanelProps) {
  if (!approval) {
    return null;
  }

  return (
    <section className="flex flex-col gap-3 rounded-3xl border border-amber-500/40 bg-amber-950/30 p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-amber-200">Approval</h2>
        <span className="rounded-full bg-amber-400/20 px-2 py-0.5 text-[10px] uppercase tracking-wide text-amber-100">
          required
        </span>
      </div>
      <div className="rounded-2xl border border-amber-500/30 bg-amber-950/40 px-3 py-2 text-sm text-amber-50">
        {approval.prompt.what}
      </div>
      <div className="text-xs text-amber-100/80">{approval.prompt.why}</div>
      <div className="text-xs text-amber-200/90">Риск: {approval.prompt.risk}</div>
      <div className="flex flex-wrap gap-2">
        {approval.required_categories.map((category) => (
          <span
            key={category}
            className="rounded-full border border-amber-400/40 bg-amber-900/50 px-2 py-0.5 text-[10px] uppercase tracking-wide text-amber-100"
          >
            {category}
          </span>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onApproveRetry}
          disabled={approving}
          className="rounded-full bg-amber-200 px-4 py-1.5 text-xs font-semibold text-amber-950 disabled:opacity-50"
        >
          {approving ? "Approving..." : "Approve + Retry"}
        </button>
        <button
          type="button"
          onClick={onDismiss}
          disabled={approving}
          className="rounded-full border border-amber-500/40 px-3 py-1.5 text-xs text-amber-100 disabled:opacity-50"
        >
          Dismiss
        </button>
      </div>
    </section>
  );
}
