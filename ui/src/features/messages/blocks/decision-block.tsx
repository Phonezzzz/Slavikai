import { DecisionPanel } from '../../../app/components/decision-panel';
import type { DecisionRespondChoice, UiDecision } from '../../../app/types';

type DecisionBlockProps = {
  decision: UiDecision;
  busy?: boolean;
  error?: string | null;
  onRespond?: (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
  ) => Promise<void> | void;
};

export function DecisionBlock({
  decision,
  busy = false,
  error = null,
  onRespond,
}: DecisionBlockProps) {
  return (
    <div className="message-decision-block">
      <DecisionPanel
        decision={decision}
        busy={busy}
        error={error}
        onRespond={(choice, editedPayload) => {
          if (!onRespond) {
            return;
          }
          void onRespond(choice, editedPayload);
        }}
      />
    </div>
  );
}
