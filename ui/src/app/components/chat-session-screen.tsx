import { PanelRight } from 'lucide-react';

import { ArtifactPanel } from './artifact-panel';
import { Canvas, type CanvasMessage, type CanvasSendPayload } from './canvas';
import type { Artifact } from './artifacts-sidebar';
import type { DecisionRespondChoice, UiDecision } from '../types';

type ChatSessionScreenProps = {
  messages: CanvasMessage[];
  pendingMessage: CanvasMessage | null;
  streamingAssistantMessage: CanvasMessage | null;
  showAssistantLoading: boolean;
  sending: boolean;
  modelLabel: string;
  statusMessage: string | null;
  longPasteToFileEnabled: boolean;
  longPasteThresholdChars: number;
  forceCanvasNext: boolean;
  artifactPanelOpen: boolean;
  artifactViewerArtifactId: string | null;
  artifacts: Artifact[];
  decision: UiDecision | null;
  decisionBusy: boolean;
  decisionError: string | null;
  onSendMessage: (payload: CanvasSendPayload) => Promise<boolean>;
  onSendFeedback: (interactionId: string, rating: 'good' | 'bad') => Promise<boolean>;
  onOpenSessionDrawer: () => void;
  onToggleForceCanvasNext: () => void;
  onDecisionRespond: (
    choice: DecisionRespondChoice,
    editedAction?: Record<string, unknown> | null,
  ) => void;
  onOpenArtifactPanel: () => void;
  onCloseArtifactPanel: () => void;
  onDownloadArtifact: (artifact: Artifact) => void;
  onDownloadAll: () => void;
};

export function ChatSessionScreen({
  messages,
  pendingMessage,
  streamingAssistantMessage,
  showAssistantLoading,
  sending,
  modelLabel,
  statusMessage,
  longPasteToFileEnabled,
  longPasteThresholdChars,
  forceCanvasNext,
  artifactPanelOpen,
  artifactViewerArtifactId,
  artifacts,
  decision,
  decisionBusy,
  decisionError,
  onSendMessage,
  onSendFeedback,
  onOpenSessionDrawer,
  onToggleForceCanvasNext,
  onDecisionRespond,
  onOpenArtifactPanel,
  onCloseArtifactPanel,
  onDownloadArtifact,
  onDownloadAll,
}: ChatSessionScreenProps) {
  return (
    <>
      <Canvas
        className="h-full"
        messages={messages}
        pendingMessage={pendingMessage}
        streamingAssistantMessage={streamingAssistantMessage}
        showAssistantLoading={showAssistantLoading}
        sending={sending}
        onSendMessage={onSendMessage}
        onSendFeedback={onSendFeedback}
        modelName={modelLabel}
        onOpenSessionDrawer={onOpenSessionDrawer}
        statusMessage={statusMessage}
        longPasteToFileEnabled={longPasteToFileEnabled}
        longPasteThresholdChars={longPasteThresholdChars}
        forceCanvasNext={forceCanvasNext}
        onToggleForceCanvasNext={onToggleForceCanvasNext}
        decision={decision}
        decisionBusy={decisionBusy}
        decisionError={decisionError}
        onDecisionRespond={onDecisionRespond}
      />

      {!artifactPanelOpen ? (
        <button
          onClick={onOpenArtifactPanel}
          className="absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-[#141418] border border-[#1f1f24] hover:border-[#2a2a30] hover:bg-[#1b1b20] flex items-center justify-center transition-all cursor-pointer shadow-lg shadow-black/30"
          title="Open Artifacts"
        >
          <PanelRight className="w-4.5 h-4.5 text-[#888]" />
        </button>
      ) : null}

      <ArtifactPanel
        isOpen={artifactPanelOpen}
        onClose={onCloseArtifactPanel}
        artifacts={artifacts}
        autoOpenArtifactId={artifactViewerArtifactId}
        onDownloadArtifact={onDownloadArtifact}
        onDownloadAll={onDownloadAll}
      />
    </>
  );
}
