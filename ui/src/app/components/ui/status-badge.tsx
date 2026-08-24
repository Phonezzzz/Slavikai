import type { ReactNode } from 'react';

import { cn } from './utils';

export type StatusTone =
  | 'neutral'
  | 'active'
  | 'waiting'
  | 'success'
  | 'error'
  | 'cancelled';

const TONE_CLASSES: Record<StatusTone, string> = {
  neutral: 'border-zinc-700 bg-zinc-800/70 text-zinc-300',
  active: 'border-indigo-500/30 bg-indigo-500/15 text-indigo-300',
  waiting: 'border-amber-500/30 bg-amber-500/15 text-amber-300',
  success: 'border-emerald-500/30 bg-emerald-500/15 text-emerald-300',
  error: 'border-rose-500/30 bg-rose-500/15 text-rose-300',
  cancelled: 'border-zinc-700 bg-zinc-800/50 text-zinc-400',
};

const DOT_CLASSES: Record<StatusTone, string> = {
  neutral: 'bg-zinc-400',
  active: 'bg-indigo-400',
  waiting: 'bg-amber-400',
  success: 'bg-emerald-400',
  error: 'bg-rose-400',
  cancelled: 'bg-zinc-500',
};

export function StatusBadge({
  tone,
  children,
  dot = true,
  className,
}: {
  tone: StatusTone;
  children: ReactNode;
  dot?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium leading-4',
        TONE_CLASSES[tone],
        className,
      )}
    >
      {dot ? <span className={cn('h-1.5 w-1.5 rounded-full', DOT_CLASSES[tone])} /> : null}
      {children}
    </span>
  );
}
