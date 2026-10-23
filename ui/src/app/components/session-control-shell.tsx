import type { SessionMode } from '../types';
import { SessionDrawer } from './session-drawer';

type SessionControlShellProps = {
  isOpen: boolean;
  onClose: () => void;
  onSaved: () => void;
  sessionId: string | null;
  sessionHeader: string;
  mode: SessionMode;
  modelLabel: string;
  selectedModelValue: string | null;
};

export function SessionControlShell({
  isOpen,
  onClose,
  onSaved,
  sessionId,
  sessionHeader,
  mode,
  modelLabel,
  selectedModelValue,
}: SessionControlShellProps) {
  return (
    <SessionDrawer
      isOpen={isOpen}
      onClose={onClose}
      onSaved={onSaved}
      sessionId={sessionId}
      sessionHeader={sessionHeader}
      mode={mode}
      modelLabel={modelLabel}
      selectedModelValue={selectedModelValue}
    />
  );
}
