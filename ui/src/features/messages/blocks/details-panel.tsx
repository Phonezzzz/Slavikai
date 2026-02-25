import type { ReactNode } from 'react';

type DetailsPanelProps = {
  open: boolean;
  children: ReactNode;
  emptyLabel?: string;
};

export function DetailsPanel({ open, children, emptyLabel = 'No details.' }: DetailsPanelProps) {
  if (!open) {
    return null;
  }
  const hasChildren = Boolean(children);
  return (
    <div className="message-details-panel">
      {hasChildren ? children : <div className="message-details-empty">{emptyLabel}</div>}
    </div>
  );
}
