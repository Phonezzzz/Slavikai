import { Settings } from './Settings';

type GlobalSettingsShellProps = {
  isOpen: boolean;
  onClose: () => void;
  onSaved: () => void;
  onRefreshModels?: () => Promise<unknown>;
  onOpenSessionControls?: () => void;
};

export function GlobalSettingsShell({
  isOpen,
  onClose,
  onSaved,
  onRefreshModels,
  onOpenSessionControls,
}: GlobalSettingsShellProps) {
  return (
    <Settings
      isOpen={isOpen}
      onClose={onClose}
      onSaved={onSaved}
      onRefreshModels={onRefreshModels}
      onOpenSessionControls={onOpenSessionControls}
    />
  );
}
