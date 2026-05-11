import { useEffect, useRef, useState } from 'react';

const MIN_EXPLORER_WIDTH = 240;
const MAX_EXPLORER_WIDTH = 420;
const MIN_ASSISTANT_WIDTH = 340;
const ASSISTANT_MAX_SCREEN_SHARE = 0.5;
const MIN_TERMINAL_HEIGHT = 140;
const MAX_TERMINAL_HEIGHT = 420;
const MIN_EDITOR_WIDTH = 420;
const ASSISTANT_RESIZER_WIDTH = 6;
const EXPLORER_RESIZER_WIDTH = 6;

type WorkspaceLayoutOptions = {
  explorerVisible: boolean;
};

export function useWorkspaceLayout({ explorerVisible }: WorkspaceLayoutOptions) {
  const [explorerWidth, setExplorerWidth] = useState(280);
  const [assistantWidth, setAssistantWidth] = useState(390);
  const [terminalHeight, setTerminalHeight] = useState(220);
  const [draggingExplorer, setDraggingExplorer] = useState(false);
  const [draggingAssistant, setDraggingAssistant] = useState(false);
  const [draggingTerminal, setDraggingTerminal] = useState(false);
  const workspaceGridRef = useRef<HTMLDivElement>(null);

  const clampAssistantWidth = (nextWidth: number): number => {
    const fallbackWidth = typeof window === 'undefined' ? 1280 : window.innerWidth;
    const gridWidth = workspaceGridRef.current?.clientWidth ?? fallbackWidth;
    const fixedColumns = explorerVisible
      ? explorerWidth + EXPLORER_RESIZER_WIDTH + ASSISTANT_RESIZER_WIDTH
      : ASSISTANT_RESIZER_WIDTH;
    const maxByEditor = Math.max(
      MIN_ASSISTANT_WIDTH,
      Math.floor(gridWidth - fixedColumns - MIN_EDITOR_WIDTH),
    );
    const maxByHalfScreen = Math.max(
      MIN_ASSISTANT_WIDTH,
      Math.floor(gridWidth * ASSISTANT_MAX_SCREEN_SHARE),
    );
    const maxAllowed = Math.min(maxByEditor, maxByHalfScreen);
    return Math.min(maxAllowed, Math.max(MIN_ASSISTANT_WIDTH, nextWidth));
  };

  useEffect(() => {
    if (!(draggingExplorer || draggingAssistant || draggingTerminal)) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      if (explorerVisible && draggingExplorer) {
        setExplorerWidth((prev) =>
          Math.min(MAX_EXPLORER_WIDTH, Math.max(MIN_EXPLORER_WIDTH, prev + event.movementX)),
        );
      }
      if (draggingAssistant) {
        setAssistantWidth((prev) => clampAssistantWidth(prev + event.movementX));
      }
      if (draggingTerminal) {
        setTerminalHeight((prev) =>
          Math.min(MAX_TERMINAL_HEIGHT, Math.max(MIN_TERMINAL_HEIGHT, prev - event.movementY)),
        );
      }
    };
    const handleUp = () => {
      setDraggingExplorer(false);
      setDraggingAssistant(false);
      setDraggingTerminal(false);
    };
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [draggingAssistant, draggingExplorer, draggingTerminal, explorerVisible]);

  useEffect(() => {
    if (!explorerVisible && draggingExplorer) {
      setDraggingExplorer(false);
    }
  }, [draggingExplorer, explorerVisible]);

  useEffect(() => {
    const syncAssistantWidth = () => {
      setAssistantWidth((prev) => clampAssistantWidth(prev));
    };
    syncAssistantWidth();
    window.addEventListener('resize', syncAssistantWidth);
    return () => {
      window.removeEventListener('resize', syncAssistantWidth);
    };
  }, [explorerVisible, explorerWidth]);

  const workspaceGridColumns = explorerVisible
    ? `${explorerWidth}px ${EXPLORER_RESIZER_WIDTH}px ${assistantWidth}px ${ASSISTANT_RESIZER_WIDTH}px minmax(${MIN_EDITOR_WIDTH}px,1fr)`
    : `${assistantWidth}px ${ASSISTANT_RESIZER_WIDTH}px minmax(${MIN_EDITOR_WIDTH}px,1fr)`;

  return {
    terminalHeight,
    workspaceGridColumns,
    workspaceGridRef,
    startExplorerResize: () => setDraggingExplorer(true),
    startAssistantResize: () => setDraggingAssistant(true),
    startTerminalResize: () => setDraggingTerminal(true),
  };
}
