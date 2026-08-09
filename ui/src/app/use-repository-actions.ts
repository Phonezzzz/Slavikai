import { useEffect, useState } from 'react';

import type { SessionTransportBridge } from './session-bridges';
import { extractErrorMessage, extractSessionIdFromPayload, parseUiDecision } from './session-payload';
import { saveLastSessionId } from './session-storage';
import type { UiDecision } from './types';

export type GitDecisionOutcome = {
  kind: 'success' | 'rejected' | 'failed';
  message: string;
};

export type RepositoryActionsResult = {
  workspaceRefreshToken: number;
  gitDecisionOutcome: GitDecisionOutcome | null;
  handleWorkspaceGithubImport: (
    repoUrl: string,
    branch?: string,
  ) => Promise<{ status: 'done' | 'pending'; message?: string | null }>;
  handleDecisionResume: (resume: unknown) => void;
  handleDecisionRejected: () => void;
};

type UseRepositoryActionsOptions = {
  sessionId: string | null;
  sessionHeader: string;
  pendingDecision: UiDecision | null;
  transportRef: React.MutableRefObject<SessionTransportBridge | null>;
  applyRuntimePayload: (payload: unknown) => void;
  loadSessions: () => Promise<unknown>;
  onSessionIdChange: (sessionId: string) => void;
  onStatusMessage: (message: string | null) => void;
};

export function useRepositoryActions({
  sessionId,
  sessionHeader,
  pendingDecision,
  transportRef,
  applyRuntimePayload,
  loadSessions,
  onSessionIdChange,
  onStatusMessage,
}: UseRepositoryActionsOptions): RepositoryActionsResult {
  const [workspaceRefreshToken, setWorkspaceRefreshToken] = useState(0);
  const [gitDecisionOutcome, setGitDecisionOutcome] = useState<GitDecisionOutcome | null>(null);

  useEffect(() => {
    setGitDecisionOutcome(null);
  }, [sessionId]);

  useEffect(() => {
    if (pendingDecision?.status !== 'pending' || !pendingDecision.context) {
      return;
    }
    const context = pendingDecision.context as Record<string, unknown>;
    if (context.source_endpoint === 'workspace.git') {
      setGitDecisionOutcome(null);
    }
  }, [pendingDecision?.id, pendingDecision?.status]);

  const handleWorkspaceGithubImport = async (
    repoUrl: string,
    branch?: string,
  ): Promise<{ status: 'done' | 'pending'; message?: string | null }> => {
    if (!sessionId) {
      throw new Error('No active session. Create chat first.');
    }
    const normalizedRepoUrl = repoUrl.trim();
    if (!normalizedRepoUrl) {
      throw new Error('Repository URL is required.');
    }
    const normalizedBranch = (branch ?? '').trim();
    const args = normalizedBranch
      ? `${normalizedRepoUrl} --branch ${normalizedBranch}`
      : normalizedRepoUrl;

    const response = await fetch('/ui/api/tools/project', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        [sessionHeader]: sessionId,
      },
      body: JSON.stringify({
        command: 'github_import',
        args,
      }),
    });
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to run GitHub import.'));
    }

    const headerSession = response.headers.get(sessionHeader);
    const payloadSession = extractSessionIdFromPayload(payload);
    const nextSession =
      (headerSession && headerSession.trim()) || payloadSession || sessionId;
    if (nextSession !== sessionId) {
      onSessionIdChange(nextSession);
    }
    saveLastSessionId(nextSession);

    transportRef.current?.applySessionPayload(payload, { applyDisplay: false });
    applyRuntimePayload(payload);
    await loadSessions();
    const decision = parseUiDecision((payload as { decision?: unknown }).decision)
      ?? pendingDecision;
    if (decision?.status === 'pending' && decision.blocking === true) {
      return {
        status: 'pending',
        message: 'Действие ожидает подтверждения в DecisionPanel (AI Assistant).',
      };
    }
    setWorkspaceRefreshToken((value) => value + 1);
    return {
      status: 'done',
      message: 'GitHub import completed.',
    };
  };

  const handleDecisionResume = (resume: unknown) => {
    if (!resume || typeof resume !== 'object') {
      onStatusMessage(null);
      return;
    }
    const payload = resume as {
      ok?: unknown;
      data?: unknown;
      source_endpoint?: unknown;
      tool_name?: unknown;
      error?: unknown;
    };
    if (payload.source_endpoint === 'workspace.git') {
      const data = payload.data && typeof payload.data === 'object'
        ? (payload.data as { message?: unknown; operation?: unknown })
        : null;
      const operation = data && typeof data.operation === 'string' && data.operation.trim()
        ? data.operation.trim()
        : 'operation';
      const resultMessage = data && typeof data.message === 'string' && data.message.trim()
        ? data.message.trim()
        : null;
      if (payload.ok === true) {
        setGitDecisionOutcome({
          kind: 'success',
          message: resultMessage && resultMessage !== 'ok'
            ? resultMessage
            : `Git ${operation} completed.`,
        });
      } else {
        setGitDecisionOutcome({
          kind: 'failed',
          message:
            typeof payload.error === 'string' && payload.error.trim()
              ? payload.error
              : resultMessage ?? `Git ${operation} failed.`,
        });
      }
      onStatusMessage(null);
      return;
    }
    if (payload.source_endpoint !== 'project.command') {
      if (payload.source_endpoint !== 'auto.run') {
        onStatusMessage(null);
        return;
      }
      const data = payload.data && typeof payload.data === 'object'
        ? (payload.data as { output?: unknown; status?: unknown })
        : null;
      const statusRaw = data && typeof data.status === 'string' ? data.status : null;
      const outputRaw = data && typeof data.output === 'string' ? data.output : null;
      if (payload.ok === true) {
        if (outputRaw && outputRaw.trim()) {
          onStatusMessage(outputRaw.trim());
        } else if (statusRaw) {
          onStatusMessage(`Auto run: ${statusRaw}`);
        } else {
          onStatusMessage('Auto run resumed.');
        }
      } else {
        onStatusMessage(
          typeof payload.error === 'string' && payload.error.trim() ? payload.error : 'Auto run failed.',
        );
      }
      return;
    }

    const toolName = typeof payload.tool_name === 'string' ? payload.tool_name : 'project';
    const data = payload.data && typeof payload.data === 'object'
      ? (payload.data as { command?: unknown; output?: unknown })
      : null;
    const command = data && typeof data.command === 'string' ? data.command : null;
    if (payload.ok === true) {
      if (command === 'github_import') {
        setWorkspaceRefreshToken((value) => value + 1);
      }
      const outputPreview =
        data && typeof data.output === 'string' && data.output.trim()
          ? data.output.trim()
          : null;
      onStatusMessage(
        outputPreview
          ? outputPreview
          : `Project command (${toolName}) completed.`,
      );
      return;
    }
    onStatusMessage(
      typeof payload.error === 'string' && payload.error.trim()
        ? payload.error
        : `Project command (${toolName}) failed.`,
    );
  };

  const handleDecisionRejected = () => {
    if (pendingDecision?.context) {
      const context = pendingDecision.context as Record<string, unknown>;
      if (context.source_endpoint === 'workspace.git') {
        setGitDecisionOutcome({
          kind: 'rejected',
          message: 'Git operation cancelled.',
        });
        return;
      }
    }
    setGitDecisionOutcome(null);
  };

  return {
    workspaceRefreshToken,
    gitDecisionOutcome,
    handleWorkspaceGithubImport,
    handleDecisionResume,
    handleDecisionRejected,
  };
}
