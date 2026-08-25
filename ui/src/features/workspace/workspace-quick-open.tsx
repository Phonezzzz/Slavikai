import { useEffect, useMemo, useRef, useState } from 'react';
import { Search, X } from 'lucide-react';

export type WorkspaceQuickOpenItem = {
  path: string;
  name: string;
  dir: string;
};

type WorkspaceQuickOpenProps = {
  open: boolean;
  query: string;
  items: WorkspaceQuickOpenItem[];
  loading: boolean;
  partial: boolean;
  onQueryChange: (value: string) => void;
  onSelect: (path: string) => void;
  onClose: () => void;
};

export function WorkspaceQuickOpen({
  open,
  query,
  items,
  loading,
  partial,
  onQueryChange,
  onSelect,
  onClose,
}: WorkspaceQuickOpenProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [cursor, setCursor] = useState(0);

  useEffect(() => {
    if (!open) {
      return;
    }
    setCursor(0);
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }, [open]);

  useEffect(() => {
    setCursor(0);
  }, [items, query]);

  const hasItems = items.length > 0;
  const safeCursor = hasItems ? Math.min(cursor, items.length - 1) : 0;

  const selectedPath = useMemo(
    () => (hasItems ? items[safeCursor]?.path ?? null : null),
    [hasItems, items, safeCursor],
  );

  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh]"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative w-[760px] max-w-[92vw] max-h-[70vh] overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl shadow-black/70"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="border-b border-zinc-800/80 px-4 py-3">
          <div className="mb-2 flex items-center justify-between gap-3 text-xs text-zinc-400">
            <span>Quick Open</span>
            <span>
              primary: <kbd className="rounded border border-zinc-700 bg-zinc-800 px-1 py-0.5 text-zinc-300">Ctrl+Space</kbd>
              {' '}
              secondary: <kbd className="rounded border border-zinc-700/80 bg-zinc-900 px-1 py-0.5 text-zinc-400">Ctrl+D</kbd>
            </span>
          </div>
          <div className="flex items-center gap-2 rounded-md border border-zinc-700/80 bg-zinc-900 px-3 py-2">
            <Search className="h-4 w-4 text-zinc-500" />
            <input
              ref={inputRef}
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Escape') {
                  event.preventDefault();
                  onClose();
                  return;
                }
                if (event.key === 'ArrowDown') {
                  event.preventDefault();
                  if (!hasItems) {
                    return;
                  }
                  setCursor((prev) => Math.min(items.length - 1, prev + 1));
                  return;
                }
                if (event.key === 'ArrowUp') {
                  event.preventDefault();
                  if (!hasItems) {
                    return;
                  }
                  setCursor((prev) => Math.max(0, prev - 1));
                  return;
                }
                if (event.key === 'Enter' && selectedPath) {
                  event.preventDefault();
                  onSelect(selectedPath);
                }
              }}
              placeholder="Search files by name or path..."
              className="flex-1 bg-transparent text-sm text-zinc-300 outline-none placeholder:text-zinc-500"
            />
            {query ? (
              <button
                onClick={() => onQueryChange('')}
                className="rounded p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
                aria-label="Clear query"
                title="Clear query"
              >
                <X className="h-4 w-4" />
              </button>
            ) : null}
          </div>
        </div>

        {partial ? (
          <div className="border-b border-amber-900/40 bg-amber-950/30 px-4 py-2 text-xs text-amber-300">
            File list is partial due to workspace tree limits.
          </div>
        ) : null}

        <div className="max-h-[48vh] overflow-y-auto" data-scrollbar="always">
          {loading ? (
            <div className="px-4 py-4 text-xs text-zinc-500">Loading file index...</div>
          ) : items.length === 0 ? (
            <div className="px-4 py-6 text-xs text-zinc-500">No files found.</div>
          ) : (
            items.map((item, index) => (
              <button
                key={item.path}
                onClick={() => onSelect(item.path)}
                onMouseEnter={() => setCursor(index)}
                className={`flex w-full items-center justify-between gap-3 border-b border-zinc-800/60 px-4 py-2.5 text-left ${
                  index === safeCursor ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-300 hover:bg-zinc-900'
                }`}
                title={item.path}
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm">{item.name}</span>
                  <span className="block truncate text-xs text-zinc-500">{item.dir || '.'}</span>
                </span>
                <span className="text-[11px] text-zinc-500">Enter</span>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
