import { useMemo, useState } from 'react';
import { Check, ChevronDown, Pencil, X } from 'lucide-react';

import type { UiDecision } from '../types';

type DecisionRespondChoice =
  | 'approve_once'
  | 'approve_session'
  | 'edit_and_approve'
  | 'edit_plan'
  | 'reject';

type DecisionPanelProps = {
  decision: UiDecision;
  busy: boolean;
  error: string | null;
  onRespond: (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
  ) => Promise<void> | void;
};

export function DecisionPanel({ decision, busy, error, onRespond }: DecisionPanelProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [editAction, setEditAction] = useState<'edit_and_approve' | 'edit_plan'>('edit_and_approve');
  const [editValue, setEditValue] = useState<string>(() =>
    JSON.stringify(decision.proposed_action ?? {}, null, 2),
  );
  const [editError, setEditError] = useState<string | null>(null);

  const supportsEdit = useMemo(() => {
    return decision.options.some(
      (option) => option.action === 'edit_and_approve' || option.action === 'edit_plan',
    );
  }, [decision.options]);

  const handleApplyEdit = () => {
    try {
      const parsed = JSON.parse(editValue) as unknown;
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        setEditError('Редактирование требует JSON-объект.');
        return;
      }
      setEditError(null);
      void onRespond(editAction, parsed as Record<string, unknown>);
      setEditMode(false);
    } catch {
      setEditError('Некорректный JSON.');
    }
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

        {supportsEdit && editMode ? (
          <div className="mt-3 space-y-2">
            <textarea
              value={editValue}
              onChange={(event) => setEditValue(event.target.value)}
              rows={6}
              className="w-full rounded-lg border border-[#2a2a30] bg-[#0f0f14] px-3 py-2 font-mono text-[12px] text-[#d7d7de] outline-none focus:border-[#3a3a44]"
              disabled={busy}
            />
            {editError ? <p className="text-xs text-rose-300">{editError}</p> : null}
          </div>
        ) : null}

        {error ? <p className="mt-3 text-xs text-rose-300">{error}</p> : null}

        <div className="mt-3 flex items-center gap-2">
          {decision.options.map((option) => {
            const action = option.action;
            const isEditAction = action === 'edit_and_approve' || action === 'edit_plan';
            const baseClass =
              action === 'approve_once' || action === 'approve_session'
                ? 'bg-emerald-600 text-white hover:bg-emerald-500'
                : 'border border-[#3a3a44] text-[#d4d4db] hover:bg-[#1b1b22]';
            const icon = action === 'reject'
              ? <X className="h-3.5 w-3.5" />
              : isEditAction
                ? <Pencil className="h-3.5 w-3.5" />
                : <Check className="h-3.5 w-3.5" />;
            const onClick = () => {
              if (isEditAction) {
                setEditAction(action);
                if (editMode && editAction === action) {
                  handleApplyEdit();
                  return;
                }
                setEditMode(true);
                return;
              }
              void onRespond(action as DecisionRespondChoice, null);
            };
            return (
              <button
                key={option.id}
                type="button"
                onClick={onClick}
                disabled={busy}
                className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-50 ${baseClass}`}
                title={option.title}
                aria-label={option.title}
              >
                {icon}
                <span>
                  {isEditAction && editMode && editAction === action ? 'Apply edit' : option.title}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
