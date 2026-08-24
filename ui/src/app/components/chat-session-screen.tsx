import { PanelRight } from 'lucide-react';

import { ArtifactPanel } from './artifact-panel';
import { Canvas, type CanvasMessage, type CanvasSendPayload } from './canvas';
import type { Artifact } from './artifacts-sidebar';
import type { DecisionRespondChoice, SessionMode, UiDecision } from '../types';

type ChatSessionScreenProps = {
  messages: CanvasMessage[];
  pendingMessage: CanvasMessage | null;
  streamingAssistantMessage: CanvasMessage | null;
  sending: boolean;
  cancelling: boolean;
  modelLabel: string;
  modelProvider: string | null;
  sessionTitle: string | null;
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
  runtimeMode: SessionMode;
  onSendMessage: (payload: CanvasSendPayload) => Promise<boolean>;
  onCancelSend: () => Promise<boolean>;
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
  sending,
  cancelling,
  modelLabel,
  modelProvider,
  sessionTitle,
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
  runtimeMode,
  onSendMessage,
  onCancelSend,
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
    <div className="relative flex h-full min-h-0 w-full overflow-hidden">
      <div className="h-full min-h-0 min-w-0 flex-1">
        <Canvas
          className="h-full min-h-0"
          messages={messages}
          pendingMessage={pendingMessage}
          streamingAssistantMessage={streamingAssistantMessage}
          sending={sending}
          cancelling={cancelling}
          onSendMessage={onSendMessage}
          onCancelSend={onCancelSend}
          onSendFeedback={onSendFeedback}
          modelName={modelLabel}
          modelProvider={modelProvider}
          sessionTitle={sessionTitle}
          onOpenSessionDrawer={onOpenSessionDrawer}
          statusMessage={statusMessage}
          longPasteToFileEnabled={longPasteToFileEnabled}
          longPasteThresholdChars={longPasteThresholdChars}
          forceCanvasNext={forceCanvasNext}
          onToggleForceCanvasNext={onToggleForceCanvasNext}
          decision={decision}
          decisionBusy={decisionBusy}
          decisionError={decisionError}
          runtimeMode={runtimeMode}
          onDecisionRespond={onDecisionRespond}
        />
      </div>

      {!artifactPanelOpen ? (
        <button
          onClick={onOpenArtifactPanel}
          className="absolute right-4 top-1/2 z-10 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-full border border-zinc-800 bg-zinc-900 shadow-lg shadow-black/30 transition-all hover:border-zinc-800 hover:bg-zinc-800"
          title="Open Artifacts"
        >
          <PanelRight className="w-4.5 h-4.5 text-zinc-400" />
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
    </div>
  );
}
