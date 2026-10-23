import type { ModeTransitionsContract, SessionMode } from '../types';
import { SessionDrawer } from './session-drawer';

type SessionControlShellProps = {
  isOpen: boolean;
  onClose: () => void;
  onSaved: () => void;
  sessionId: string | null;
  sessionHeader: string;
  mode: SessionMode;
  modeTransitions: ModeTransitionsContract | null;
  modeBusy: boolean;
  onChangeMode: (mode: SessionMode) => Promise<void>;
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
  modeTransitions,
  modeBusy,
  onChangeMode,
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
      modeTransitions={modeTransitions}
      modeBusy={modeBusy}
      onChangeMode={onChangeMode}
      modelLabel={modelLabel}
      selectedModelValue={selectedModelValue}
    />
  );
}
