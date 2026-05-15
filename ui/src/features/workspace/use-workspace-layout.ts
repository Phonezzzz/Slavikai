import { useEffect, useRef, useState } from 'react';

const MIN_EXPLORER_WIDTH = 240;
const MAX_EXPLORER_WIDTH = 420;
const MIN_TERMINAL_HEIGHT = 140;
const MAX_TERMINAL_HEIGHT = 420;
const MIN_EDITOR_WIDTH = 420;
const EXPLORER_RESIZER_WIDTH = 6;

type WorkspaceLayoutOptions = {
  explorerVisible: boolean;
};

export function useWorkspaceLayout({ explorerVisible }: WorkspaceLayoutOptions) {
  const [explorerWidth, setExplorerWidth] = useState(280);
  const [terminalHeight, setTerminalHeight] = useState(220);
  const [draggingExplorer, setDraggingExplorer] = useState(false);
  const [draggingTerminal, setDraggingTerminal] = useState(false);
  const workspaceGridRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!(draggingExplorer || draggingTerminal)) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      if (explorerVisible && draggingExplorer) {
        setExplorerWidth((prev) =>
          Math.min(MAX_EXPLORER_WIDTH, Math.max(MIN_EXPLORER_WIDTH, prev + event.movementX)),
        );
      }
      if (draggingTerminal) {
        setTerminalHeight((prev) =>
          Math.min(MAX_TERMINAL_HEIGHT, Math.max(MIN_TERMINAL_HEIGHT, prev - event.movementY)),
        );
      }
    };
    const handleUp = () => {
      setDraggingExplorer(false);
      setDraggingTerminal(false);
    };
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [draggingExplorer, draggingTerminal, explorerVisible]);

  useEffect(() => {
    if (!explorerVisible && draggingExplorer) {
      setDraggingExplorer(false);
    }
  }, [draggingExplorer, explorerVisible]);

  const filesTabColumns = explorerVisible
    ? `${explorerWidth}px ${EXPLORER_RESIZER_WIDTH}px minmax(${MIN_EDITOR_WIDTH}px,1fr)`
    : `minmax(${MIN_EDITOR_WIDTH}px,1fr)`;

  return {
    terminalHeight,
    filesTabColumns,
    workspaceGridRef,
    startExplorerResize: () => setDraggingExplorer(true),
    startTerminalResize: () => setDraggingTerminal(true),
  };
}
