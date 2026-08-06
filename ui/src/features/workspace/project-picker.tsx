import { useEffect, useRef, useState } from 'react';
import { Folder, FolderOpen, X, ArrowUp, ChevronRight } from 'lucide-react';
import {
  fetchWorkspaceBrowse,
  type BrowseResult,
} from './workspace-api';
import { compactPath } from './workspace-helpers';

type ProjectPickerProps = {
  sessionId: string | null;
  sessionHeader: string;
  workspaceRoot: string;
  loading: boolean;
  onApplyRoot: (rootPath: string) => void;
  onClose: () => void;
};

const RECENT_ROOTS_KEY = 'slavik.project-picker.recent-roots';
const MAX_RECENT_ROOTS = 10;

const loadRecentRoots = (): string[] => {
  try {
    const raw = localStorage.getItem(RECENT_ROOTS_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed.filter((r): r is string => typeof r === 'string').slice(0, MAX_RECENT_ROOTS);
    }
  } catch {
    // ignore
  }
  return [];
};

const saveRecentRoot = (rootPath: string): void => {
  const current = loadRecentRoots();
  const filtered = current.filter((r) => r !== rootPath);
  const next = [rootPath, ...filtered].slice(0, MAX_RECENT_ROOTS);
  try {
    localStorage.setItem(RECENT_ROOTS_KEY, JSON.stringify(next));
  } catch {
    // ignore
  }
};

export function ProjectPicker({
  sessionId,
  sessionHeader,
  workspaceRoot,
  loading,
  onApplyRoot,
  onClose,
}: ProjectPickerProps) {
  const [browsePath, setBrowsePath] = useState(workspaceRoot || '');
  const [browseResult, setBrowseResult] = useState<BrowseResult | null>(null);
  const [browseError, setBrowseError] = useState<string | null>(null);
  const [browseLoading, setBrowseLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [recentRoots] = useState<string[]>(() => loadRecentRoots());
  const listRef = useRef<HTMLDivElement>(null);

  const requestHeaders: Record<string, string> = sessionId
    ? { [sessionHeader]: sessionId }
    : {};

  const loadBrowse = async (path: string) => {
    if (!sessionId) return;
    setBrowseLoading(true);
    setBrowseError(null);
    try {
      const result = await fetchWorkspaceBrowse(path, requestHeaders);
      setBrowseResult(result);
      setBrowsePath(result.path);
      setSelectedIndex(0);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Browse failed.';
      setBrowseError(message);
    } finally {
      setBrowseLoading(false);
    }
  };

  useEffect(() => {
    void loadBrowse(browsePath || '');
  }, []);

  const filteredEntries = browseResult
    ? browseResult.entries.filter((e) =>
        searchQuery
          ? e.name.toLowerCase().includes(searchQuery.toLowerCase())
          : true,
      )
    : [];

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, Math.max(0, filteredEntries.length - 1)));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (event.key === 'Enter') {
      event.preventDefault();
      const entry = filteredEntries[selectedIndex];
      if (entry) {
        void loadBrowse(entry.path);
      }
    } else if (event.key === 'Escape') {
      onClose();
    }
  };

  useEffect(() => {
    const sel = listRef.current?.querySelector('[data-selected="true"]');
    if (sel) {
      sel.scrollIntoView({ block: 'nearest' });
    }
  }, [selectedIndex]);

  const handleApplyCurrent = () => {
    const root = browsePath.trim();
    if (root) {
      saveRecentRoot(root);
      onApplyRoot(root);
    }
  };

  const handleRecentClick = (recentPath: string) => {
    saveRecentRoot(recentPath);
    onApplyRoot(recentPath);
  };

  const navigateToParent = () => {
    if (browseResult?.parent) {
      void loadBrowse(browseResult.parent);
    }
  };

  const navigateBreadcrumb = (path: string) => {
    void loadBrowse(path);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg overflow-hidden rounded-xl border border-[#2a2a31] bg-[#111117] shadow-2xl"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={handleKeyDown}
        role="dialog"
        aria-label="Project Picker"
      >
        <div className="flex items-center justify-between border-b border-[#1f1f24] px-4 py-3">
          <h2 className="text-sm font-semibold text-[#d0d0d8]">Open Project</h2>
          <button
            onClick={onClose}
            className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-[#2a2a31] text-[#8f8f98] hover:bg-[#1a1a21]"
            aria-label="Close"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="border-b border-[#1f1f24] px-4 py-2">
          <input
            value={searchQuery}
            onChange={(event) => {
              setSearchQuery(event.target.value);
              setSelectedIndex(0);
            }}
            placeholder="Search directories..."
            className="w-full rounded-md border border-[#2a2a31] bg-[#0d0d12] px-3 py-1.5 text-[12px] text-[#d0d0d8] outline-none placeholder:text-[#555]"
          />
        </div>

        {browseResult && browseResult.breadcrumbs.length > 1 ? (
          <div className="border-b border-[#1f1f24] px-4 py-1.5 flex items-center gap-1 overflow-x-auto">
            {browseResult.breadcrumbs.map((crumb, idx) => (
              <span key={crumb.path} className="flex items-center gap-1 shrink-0">
                {idx > 0 ? (
                  <ChevronRight className="h-3 w-3 text-[#555]" />
                ) : null}
                <button
                  onClick={() => navigateBreadcrumb(crumb.path)}
                  className="text-[11px] text-[#8f8f98] hover:text-[#d0d0d8] truncate max-w-[120px]"
                >
                  {crumb.name}
                </button>
              </span>
            ))}
          </div>
        ) : null}

        <div className="border-b border-[#1f1f24] px-4 py-1.5 flex items-center justify-between">
          <span className="text-[11px] text-[#666]">
            {browseLoading
              ? 'Loading...'
              : browsePath
                ? compactPath(browsePath, 50)
                : 'Select directory'}
          </span>
          {browseResult?.parent ? (
            <button
              onClick={navigateToParent}
              className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#0d0d12] px-2 py-0.5 text-[11px] text-[#bdbdc6] hover:bg-[#1a1a21]"
            >
              <ArrowUp className="h-3 w-3" />
              Parent
            </button>
          ) : null}
        </div>

        {recentRoots.length > 0 && !searchQuery ? (
          <div className="border-b border-[#1f1f24] px-4 py-2">
            <div className="text-[10px] uppercase tracking-wider text-[#555] mb-1">Recent</div>
            <div className="flex flex-wrap gap-1">
              {recentRoots.map((recent) => (
                <button
                  key={recent}
                  onClick={() => handleRecentClick(recent)}
                  className="rounded-md border border-[#2a2a31] bg-[#0d0d12] px-2 py-0.5 text-[11px] text-[#8f8f98] hover:bg-[#1a1a21] truncate max-w-[180px]"
                  title={recent}
                >
                  {compactPath(recent, 30)}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        <div ref={listRef} className="max-h-[280px] overflow-auto" data-scrollbar="always">
          {browseError ? (
            <div className="px-4 py-3 text-[12px] text-red-400">{browseError}</div>
          ) : browseLoading ? (
            <div className="px-4 py-3 text-[12px] text-[#666]">Loading...</div>
          ) : filteredEntries.length === 0 ? (
            <div className="px-4 py-3 text-[12px] text-[#666]">
              {searchQuery ? 'No matching directories.' : 'Empty directory.'}
            </div>
          ) : (
            filteredEntries.map((entry, idx) => {
              const isSelected = idx === selectedIndex;
              return (
                <button
                  key={entry.path}
                  onClick={() => {
                    void loadBrowse(entry.path);
                  }}
                  onDoubleClick={handleApplyCurrent}
                  data-selected={isSelected ? 'true' : 'false'}
                  className={`flex w-full items-center gap-2 px-4 py-1.5 text-left text-[12px] transition-colors ${
                    isSelected
                      ? 'bg-[#1b1b22] text-[#d6d6de]'
                      : 'text-[#a4a4ad] hover:bg-[#15151a]'
                  }`}
                >
                  {browsePath === entry.path ? (
                    <FolderOpen className="h-3.5 w-3.5 shrink-0 text-[#f59e0b]" />
                  ) : (
                    <Folder className="h-3.5 w-3.5 shrink-0 text-[#f59e0b]" />
                  )}
                  <span className="truncate">{entry.name}</span>
                </button>
              );
            })
          )}
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-[#1f1f24] px-4 py-3">
          <button
            onClick={onClose}
            className="rounded-md border border-[#2a2a31] bg-[#111117] px-3 py-1.5 text-[12px] text-[#b3b3bc] hover:bg-[#1a1a21]"
          >
            Cancel
          </button>
          <button
            onClick={handleApplyCurrent}
            disabled={loading || !browsePath}
            className="rounded-md border border-[#3a3a46] bg-[#1a1a22] px-3 py-1.5 text-[12px] text-[#d4d4dd] hover:bg-[#22223a] disabled:opacity-50"
          >
            {loading ? 'Applying...' : 'Open Project'}
          </button>
        </div>
      </div>
    </div>
  );
}
