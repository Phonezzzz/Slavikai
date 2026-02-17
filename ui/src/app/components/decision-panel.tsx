import { useState } from 'react';
import { Check, ChevronDown, X } from 'lucide-react';

import type { UiDecision } from '../types';

type DecisionRespondChoice = 'approve' | 'reject';

type DecisionPanelProps = {
  decision: UiDecision;
  busy: boolean;
  error: string | null;
  onRespond: (
    choice: DecisionRespondChoice,
  ) => Promise<void> | void;
};

export function DecisionPanel({ decision, busy, error, onRespond }: DecisionPanelProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);

  const handleApprove = () => {
    void onRespond('approve');
  };

  const handleReject = () => {
    void onRespond('reject');
  };

  return (
    <div className="border-t border-[#1f1f24] bg-[#101014] px-4 py-3">
      <div className="mx-auto max-w-3xl rounded-xl border border-[#2a2a30] bg-[#141418] px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-[#e4e4e7]">Требуется подтверждение</h3>
            <p className="mt-1 text-xs text-[#b4b4bd]">{decision.summary}</p>
            <p className="mt-1 text-xs text-[#8f8f98]">Причина: {decision.reason}</p>
          </div>
          <button
            type="button"
            onClick={() => setDetailsOpen((prev) => !prev)}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a30] px-2 py-1 text-xs text-[#b4b4bd] hover:bg-[#1b1b22]"
            title="Show details"
            aria-label="Show details"
          >
            <span>Details</span>
            <ChevronDown className={`h-3.5 w-3.5 transition-transform ${detailsOpen ? 'rotate-180' : ''}`} />
          </button>
        </div>

        {detailsOpen ? (
          <div className="mt-3 rounded-lg border border-[#23232a] bg-[#0f0f14] p-3">
            <pre className="overflow-auto text-[11px] leading-relaxed text-[#c9c9d2]">
              {JSON.stringify(decision.proposed_action ?? {}, null, 2)}
            </pre>
          </div>
        ) : null}

        {error ? <p className="mt-3 text-xs text-rose-300">{error}</p> : null}

        <div className="mt-3 flex items-center gap-2">
          <button
            type="button"
            onClick={handleApprove}
            disabled={busy}
            className="inline-flex items-center gap-1.5 rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
            title="Approve"
            aria-label="Approve"
          >
            <Check className="h-3.5 w-3.5" />
            <span>Approve</span>
          </button>
          <button
            type="button"
            onClick={handleReject}
            disabled={busy}
            className="inline-flex items-center gap-1.5 rounded-md border border-[#3a3a44] px-3 py-1.5 text-xs font-medium text-[#d4d4db] hover:bg-[#1b1b22] disabled:cursor-not-allowed disabled:opacity-50"
            title="Reject"
            aria-label="Reject"
          >
            <X className="h-3.5 w-3.5" />
            <span>Reject</span>
          </button>
        </div>
      </div>
    </div>
  );
}
