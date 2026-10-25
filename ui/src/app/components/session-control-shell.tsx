import type { SessionMode } from '../types';
import { SessionDrawer } from './session-drawer';

type SessionControlShellProps = {
  isOpen: boolean;
  onClose: () => void;
  onSaveSettled?: (sessionId: string) => void;
  sessionId: string | null;
  sessionHeader: string;
  mode: SessionMode;
  modelLabel: string;
  selectedModelValue: string | null;
};

export function SessionControlShell({
  isOpen,
  onClose,
  onSaveSettled,
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
      onSaveSettled={onSaveSettled}
      sessionId={sessionId}
      sessionHeader={sessionHeader}
      mode={mode}
      modelLabel={modelLabel}
      selectedModelValue={selectedModelValue}
    />
  );
}
