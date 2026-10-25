import { Settings } from './Settings';

type GlobalSettingsShellProps = {
  isOpen: boolean;
  onClose: () => void;
  onSaved: () => void;
  onOpenSessionControls?: () => void;
};

export function GlobalSettingsShell({
  isOpen,
  onClose,
  onSaved,
  onOpenSessionControls,
}: GlobalSettingsShellProps) {
  return (
    <Settings
      isOpen={isOpen}
      onClose={onClose}
      onSaved={onSaved}
      onOpenSessionControls={onOpenSessionControls}
    />
  );
}
