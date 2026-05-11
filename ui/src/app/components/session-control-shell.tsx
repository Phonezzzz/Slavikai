import type { ModeTransitionsContract, ProviderModels, SessionMode } from '../types';
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
  providerModels: ProviderModels[];
  selectedModelValue: string | null;
  modelsLoading: boolean;
  savingModel: boolean;
  onLoadProviderModels: (provider: string) => Promise<ProviderModels | null>;
  onStartLocalOllama: () => Promise<ProviderModels | null>;
  onSelectModel: (provider: string, model: string) => void;
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
  providerModels,
  selectedModelValue,
  modelsLoading,
  savingModel,
  onLoadProviderModels,
  onStartLocalOllama,
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
      modeTransitions={modeTransitions}
      modeBusy={modeBusy}
      onChangeMode={onChangeMode}
      modelLabel={modelLabel}
      providerModels={providerModels}
      selectedModelValue={selectedModelValue}
      modelsLoading={modelsLoading}
      savingModel={savingModel}
      onLoadProviderModels={onLoadProviderModels}
      onStartLocalOllama={onStartLocalOllama}
      onSelectModel={onSelectModel}
    />
  );
}
