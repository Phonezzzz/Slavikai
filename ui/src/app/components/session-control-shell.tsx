import type { SessionMode } from '../types';
import { SessionDrawer } from './session-drawer';

type SessionModelOption = {
  value: string;
  label: string;
  provider: string;
  model: string;
  disabled?: boolean;
};

type SessionControlShellProps = {
  isOpen: boolean;
  onClose: () => void;
  onSaved: () => void;
  sessionId: string | null;
  sessionHeader: string;
  mode: SessionMode;
  modeBusy: boolean;
  onChangeMode: (mode: SessionMode) => Promise<void>;
  modelLabel: string;
  modelOptions: SessionModelOption[];
  selectedModelValue: string | null;
  modelsLoading: boolean;
  savingModel: boolean;
  onSelectModel: (provider: string, model: string) => void;
};

export function SessionControlShell({
  isOpen,
  onClose,
  onSaved,
  sessionId,
  sessionHeader,
  mode,
  modeBusy,
  onChangeMode,
  modelLabel,
  modelOptions,
  selectedModelValue,
  modelsLoading,
  savingModel,
  onSelectModel,
}: SessionControlShellProps) {
  return (
    <SessionDrawer
      isOpen={isOpen}
      onClose={onClose}
      onSaved={onSaved}
      sessionId={sessionId}
      sessionHeader={sessionHeader}
      mode={mode}
      modeBusy={modeBusy}
      onChangeMode={onChangeMode}
      modelLabel={modelLabel}
      modelOptions={modelOptions}
      selectedModelValue={selectedModelValue}
      modelsLoading={modelsLoading}
      savingModel={savingModel}
      onSelectModel={onSelectModel}
    />
  );
}
