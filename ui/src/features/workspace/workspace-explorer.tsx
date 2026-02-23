import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, ChevronRight, Folder, FolderOpen } from 'lucide-react';

import {
  fileIcon,
  flattenWorkspaceTree,
  type WorkspaceNode,
  type WorkspaceTreeMeta,
} from './workspace-helpers';

const ROW_HEIGHT = 28;
const OVERSCAN = 10;

type WorkspaceExplorerProps = {
  tree: WorkspaceNode[];
  treeLoading: boolean;
  treeError: string | null;
  loadingTreePaths: Set<string>;
  treeMeta: WorkspaceTreeMeta | null;
  expandedNodes: Set<string>;
  activePath: string | null;
  activeExplorerPath: string | null;
  onToggleNode: (node: WorkspaceNode, key: string, expanded: boolean) => void;
  onSelectPath: (path: string | null) => void;
  onOpenFile: (path: string) => void;
  onCreateFile: () => void;
  onRenamePath: (path: string | null) => void;
  onMovePath: (path: string | null) => void;
  onDeletePath: (path: string | null) => void;
};

export function WorkspaceExplorer({
  tree,
  treeLoading,
  treeError,
  loadingTreePaths,
  treeMeta,
  expandedNodes,
  activePath,
  activeExplorerPath,
  onToggleNode,
  onSelectPath,
  onOpenFile,
  onCreateFile,
  onRenamePath,
  onMovePath,
  onDeletePath,
}: WorkspaceExplorerProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) {
      return;
    }
    const updateHeight = () => {
      setViewportHeight(element.clientHeight);
    };
    updateHeight();
    const resizeObserver = new ResizeObserver(() => updateHeight());
    resizeObserver.observe(element);
    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  const flatRows = useMemo(
    () => flattenWorkspaceTree(tree, expandedNodes),
    [expandedNodes, tree],
  );
  const totalRows = flatRows.length;
  const visibleCount = Math.max(
    1,
    Math.ceil(viewportHeight / ROW_HEIGHT) + OVERSCAN * 2,
  );
  const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
  const endIndex = Math.min(totalRows, startIndex + visibleCount);
  const rows = flatRows.slice(startIndex, endIndex);
  const topSpacerHeight = startIndex * ROW_HEIGHT;
  const bottomSpacerHeight = Math.max(0, (totalRows - endIndex) * ROW_HEIGHT);

  const renderRow = (
    node: WorkspaceNode,
    key: string,
    depth: number,
    parentKey: string,
  ): JSX.Element => {
    if (node.type === 'dir') {
      const expanded = expandedNodes.has(key);
      const selected = activeExplorerPath === node.path;
      const hasChildren = node.hasChildren === true || (node.children?.length ?? 0) > 0;
      const loading = loadingTreePaths.has(node.path);
      return (
        <button
          key={key}
          onClick={() => {
            onSelectPath(node.path);
            onToggleNode(node, key, expanded);
          }}
          className={`flex h-[28px] w-full items-center gap-1.5 px-2 text-left text-[12px] hover:bg-[#15151a] ${
            selected ? 'bg-[#1b1b22] text-[#d6d6de]' : 'text-[#a4a4ad]'
          }`}
          style={{ paddingLeft: `${8 + depth * 14}px` }}
          data-parent={parentKey}
        >
          {hasChildren ? (
            expanded ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )
          ) : (
            <span className="inline-block h-3.5 w-3.5" />
          )}
          {expanded ? (
            <FolderOpen className="h-3.5 w-3.5 text-[#f59e0b]" />
          ) : (
            <Folder className="h-3.5 w-3.5 text-[#f59e0b]" />
          )}
          <span className="truncate">{node.name}</span>
          {node.childrenTruncated ? (
            <span className="text-[#f59e0b]" title="Список детей директории обрезан лимитами">
              …
            </span>
          ) : null}
          {loading ? (
            <span className="text-[#7d7d86]" title="Загрузка директории">
              ...
            </span>
          ) : null}
        </button>
      );
    }

    const path = node.path;
    const isActive = activePath === path;
    const selected = activeExplorerPath === path;
    return (
      <button
        key={key}
        onClick={() => {
          onSelectPath(path);
          onOpenFile(path);
        }}
        className={`flex h-[28px] w-full items-center gap-1.5 px-2 text-left text-[12px] transition-colors ${
          isActive || selected
            ? 'bg-[#1b1b22] text-[#d6d6de]'
            : 'text-[#9a9aa3] hover:bg-[#15151a]'
        }`}
        style={{ paddingLeft: `${28 + depth * 14}px` }}
      >
        {fileIcon(path)}
        <span className="truncate">{node.name}</span>
      </button>
    );
  };

  const treeTruncated = treeMeta?.truncated === true;
  const treeTruncatedTitle = treeMeta?.truncatedReasons.join(', ') ?? '';

  return (
    <aside className="min-h-0 border-r border-[#1f1f24] bg-[#0d0d11] flex flex-col overflow-hidden">
      <div className="h-9 flex items-center justify-between gap-2 px-3 border-b border-[#1f1f24]">
        <span className="text-[11px] uppercase tracking-wider text-[#686873]">Explorer</span>
        <div className="flex items-center gap-1 text-[11px]">
          <button
            onClick={onCreateFile}
            className="rounded border border-[#2a2a31] bg-[#121217] px-1.5 py-0.5 text-[#bdbdc6] hover:bg-[#1a1a21]"
            title="Create file"
          >
            New
          </button>
          <button
            onClick={() => onRenamePath(activeExplorerPath)}
            className="rounded border border-[#2a2a31] bg-[#121217] px-1.5 py-0.5 text-[#bdbdc6] hover:bg-[#1a1a21] disabled:opacity-50"
            disabled={!activeExplorerPath}
            title="Rename selected path"
          >
            Rename
          </button>
          <button
            onClick={() => onMovePath(activeExplorerPath)}
            className="rounded border border-[#2a2a31] bg-[#121217] px-1.5 py-0.5 text-[#bdbdc6] hover:bg-[#1a1a21] disabled:opacity-50"
            disabled={!activeExplorerPath}
            title="Move selected path"
          >
            Move
          </button>
          <button
            onClick={() => onDeletePath(activeExplorerPath)}
            className="rounded border border-[#2a2a31] bg-[#121217] px-1.5 py-0.5 text-[#ef8f8f] hover:bg-[#2a1717] disabled:opacity-50"
            disabled={!activeExplorerPath}
            title="Delete selected path"
          >
            Del
          </button>
        </div>
      </div>
      {treeTruncated ? (
        <div
          className="border-b border-[#2a2212] bg-[#1b1308] px-3 py-1 text-[11px] text-[#f2c27f]"
          title={treeTruncatedTitle}
        >
          Показана часть дерева из-за лимитов производительности.
        </div>
      ) : null}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-auto"
        data-scrollbar="always"
        onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
      >
        {treeLoading ? (
          <div className="px-3 py-2 text-[12px] text-[#777]">Loading tree...</div>
        ) : treeError ? (
          <div className="px-3 py-2 text-[12px] text-red-400">{treeError}</div>
        ) : flatRows.length === 0 ? (
          <div className="px-3 py-2 text-[12px] text-[#777]">Workspace is empty.</div>
        ) : (
          <div>
            <div style={{ height: `${topSpacerHeight}px` }} />
            {rows.map((row) =>
              renderRow(row.node, row.key, row.depth, row.parentKey),
            )}
            <div style={{ height: `${bottomSpacerHeight}px` }} />
          </div>
        )}
      </div>
    </aside>
  );
}
