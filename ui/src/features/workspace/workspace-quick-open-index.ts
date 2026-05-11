import type { WorkspaceNode } from './workspace-helpers';
import type { WorkspaceQuickOpenItem } from './workspace-quick-open';

export const QUICK_OPEN_MAX_RESULTS = 120;
export const QUICK_OPEN_MAX_RECENT = 24;

export type QuickOpenIndexCache = {
  rootKey: string;
  items: WorkspaceQuickOpenItem[];
  partial: boolean;
  loadedAt: number;
};

export function collectQuickOpenItems(nodes: WorkspaceNode[]): WorkspaceQuickOpenItem[] {
  const output: WorkspaceQuickOpenItem[] = [];
  const walk = (items: WorkspaceNode[]) => {
    for (const node of items) {
      if (node.type === 'file') {
        const path = node.path?.trim() ?? '';
        if (!path) {
          continue;
        }
        const slash = path.lastIndexOf('/');
        const name = slash >= 0 ? path.slice(slash + 1) : path;
        const dir = slash >= 0 ? path.slice(0, slash) : '';
        output.push({ path, name, dir });
        continue;
      }
      if (node.children && node.children.length > 0) {
        walk(node.children);
      }
    }
  };
  walk(nodes);
  output.sort((a, b) => {
    const byName = a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
    if (byName !== 0) {
      return byName;
    }
    return a.path.localeCompare(b.path, undefined, { sensitivity: 'base' });
  });
  return output;
}

export function filterQuickOpenItems(
  items: WorkspaceQuickOpenItem[],
  query: string,
  recentPaths: string[],
  limit = QUICK_OPEN_MAX_RESULTS,
): WorkspaceQuickOpenItem[] {
  const rawQuery = query.trim().toLowerCase();
  if (!rawQuery) {
    return items.slice(0, limit);
  }
  const recentBoost = new Set(recentPaths);
  const scored = items
    .map((item) => {
      const name = item.name.toLowerCase();
      const path = item.path.toLowerCase();
      let score = -1;
      if (name === rawQuery) {
        score = 400;
      } else if (name.startsWith(rawQuery)) {
        score = 300;
      } else if (name.includes(rawQuery)) {
        score = 200;
      } else if (path.includes(rawQuery)) {
        score = 120;
      }
      if (score < 0) {
        return null;
      }
      if (recentBoost.has(item.path)) {
        score += 35;
      }
      return { item, score };
    })
    .filter((entry): entry is { item: WorkspaceQuickOpenItem; score: number } => entry !== null)
    .sort((a, b) => {
      if (b.score !== a.score) {
        return b.score - a.score;
      }
      return a.item.path.localeCompare(b.item.path, undefined, { sensitivity: 'base' });
    });
  return scored.slice(0, limit).map((entry) => entry.item);
}

export function nextRecentWorkspacePaths(
  previous: string[],
  path: string,
  limit = QUICK_OPEN_MAX_RECENT,
): string[] {
  const normalized = path.trim();
  if (!normalized) {
    return previous;
  }
  const next = [normalized, ...previous.filter((item) => item !== normalized)];
  return next.slice(0, limit);
}
