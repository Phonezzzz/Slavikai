import { ChevronDown, ChevronRight, Folder, FolderOpen } from 'lucide-react';

import { fileIcon, nodeKey, type WorkspaceNode } from './workspace-helpers';

type WorkspaceExplorerProps = {
  tree: WorkspaceNode[];
  treeLoading: boolean;
  treeError: string | null;
  expandedNodes: Set<string>;
  activePath: string | null;
  onToggleNode: (key: string) => void;
  onOpenFile: (path: string) => void;
};

export function WorkspaceExplorer({
  tree,
  treeLoading,
  treeError,
  expandedNodes,
  activePath,
  onToggleNode,
  onOpenFile,
}: WorkspaceExplorerProps) {
  const renderNode = (node: WorkspaceNode, parent: string, depth: number): JSX.Element => {
    const key = nodeKey(node, parent);
    if (node.type === 'dir') {
      const expanded = expandedNodes.has(key);
      return (
        <div key={key}>
          <button
            onClick={() => onToggleNode(key)}
            className="flex w-full items-center gap-1.5 px-2 py-1.5 text-left text-[12px] text-[#a4a4ad] hover:bg-[#15151a]"
            style={{ paddingLeft: `${8 + depth * 14}px` }}
          >
            {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            {expanded ? <FolderOpen className="h-3.5 w-3.5 text-[#f59e0b]" /> : <Folder className="h-3.5 w-3.5 text-[#f59e0b]" />}
            <span className="truncate">{node.name}</span>
          </button>
          {expanded && (node.children ?? []).map((child) => renderNode(child, key, depth + 1))}
        </div>
      );
    }

    const path = node.path ?? node.name;
    const isActive = activePath === path;
    return (
      <button
        key={key}
        onClick={() => onOpenFile(path)}
        className={`flex w-full items-center gap-1.5 px-2 py-1.5 text-left text-[12px] transition-colors ${
          isActive ? 'bg-[#1b1b22] text-[#d6d6de]' : 'text-[#9a9aa3] hover:bg-[#15151a]'
        }`}
        style={{ paddingLeft: `${28 + depth * 14}px` }}
      >
        {fileIcon(path)}
        <span className="truncate">{node.name}</span>
      </button>
    );
  };

  return (
    <aside className="min-h-0 border-r border-[#1f1f24] bg-[#0d0d11] flex flex-col overflow-hidden">
      <div className="h-9 flex items-center px-3 border-b border-[#1f1f24] text-[11px] uppercase tracking-wider text-[#686873]">
        Explorer
      </div>
      <div className="flex-1 min-h-0 overflow-auto" data-scrollbar="always">
        {treeLoading ? (
          <div className="px-3 py-2 text-[12px] text-[#777]">Loading tree...</div>
        ) : treeError ? (
          <div className="px-3 py-2 text-[12px] text-red-400">{treeError}</div>
        ) : tree.length === 0 ? (
          <div className="px-3 py-2 text-[12px] text-[#777]">Workspace is empty.</div>
        ) : (
          tree.map((node) => renderNode(node, '', 0))
        )}
      </div>
    </aside>
  );
}
