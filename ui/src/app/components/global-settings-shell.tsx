import { Settings } from './Settings';

type GlobalSettingsShellProps = {
  isOpen: boolean;
  onClose: () => void;
  onSaved: () => void;
};

export function GlobalSettingsShell({
  isOpen,
  onClose,
  onSaved,
}: GlobalSettingsShellProps) {
  return (
    <Settings
      isOpen={isOpen}
      onClose={onClose}
      onSaved={onSaved}
    />
  );
}
