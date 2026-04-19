import { useEffect, useRef, useState, type MutableRefObject } from 'react';

import type { ComposerUiSettings } from './session-payload';
import {
  extractErrorMessage,
  extractSessionIdFromPayload,
  parseAutoState,
  parseDecisionResumeWorkspaceRoot,
  parseMessages,
  parseModeTransitions,
  parsePlanEnvelope,
  parseSelectedModel,
  parseSessionArtifacts,
  parseSessionFiles,
  parseSessionMode,
  parseSessionOutput,
  parseTaskExecution,
  parseUiDecision,
  parseWorkspaceRoot,
} from './session-payload';
import {
  DEFAULT_SESSION_SECURITY_SUMMARY,
  loadSessionSecuritySummary,
  type SessionSecuritySummary,
} from './session-security';
import {
  loadLastModel,
  loadLastSessionId,
  saveLastModel,
  saveLastSessionId,
} from './session-storage';
import type { SessionTransportBridge } from './session-bridges';
import type {
  AutoState,
  DecisionRespondChoice,
  ModeTransitionsContract,
  PlanEnvelope,
  ProviderModels,
  SelectedModel,
  SessionMode,
  SessionSummary,
  TaskExecutionState,
  UiDecision,
} from './types';

type WorkflowSnapshot = {
  mode: SessionMode;
  activePlan: PlanEnvelope | null;
  activeTask: TaskExecutionState | null;
  autoState: AutoState | null;
  modeTransitions: ModeTransitionsContract | null;
};

type UseSessionRuntimeControllerOptions = {
  sessionHeader: string;
  providerModels: ProviderModels[];
  loadSessions: () => Promise<SessionSummary[]>;
  loadModels: () => Promise<ProviderModels[]>;
  loadComposerSettings: () => Promise<ComposerUiSettings>;
  onStatusMessage: (message: string | null) => void;
  transportRef: MutableRefObject<SessionTransportBridge | null>;
};

export type SessionRuntimeControllerResult = {
  selectedConversation: string | null;
  selectedModel: SelectedModel | null;
  savingModel: boolean;
  sessionMode: SessionMode;
  activePlan: PlanEnvelope | null;
  activeTask: TaskExecutionState | null;
  autoState: AutoState | null;
  modeTransitions: ModeTransitionsContract | null;
  pendingDecision: UiDecision | null;
  decisionBusy: boolean;
  decisionError: string | null;
  sessionSecuritySummary: SessionSecuritySummary;
  workspaceRoot: string;
  modeBusy: boolean;
  modeError: string | null;
  handleSelectConversation: (sessionId: string) => Promise<void>;
  handleCreateConversation: () => Promise<void>;
  handleDeleteConversation: (sessionId: string) => Promise<void>;
  handleSetModel: (provider: string, model: string) => Promise<boolean>;
  handleChangeMode: (mode: SessionMode) => Promise<void>;
  handlePlanDraft: (goal: string) => Promise<void>;
  handlePlanApprove: () => Promise<void>;
  handlePlanExecute: () => Promise<void>;
  handlePlanCancel: () => Promise<void>;
  handleDecisionRespond: (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
    onResume?: (resume: unknown) => void,
  ) => Promise<void>;
  applyWorkspaceRoot: (workspaceRoot: string) => void;
  refreshSessionSecuritySummary: () => Promise<void>;
  applyRuntimePayload: (payload: unknown) => void;
};

const isModelAvailable = (model: SelectedModel, providers: ProviderModels[]) =>
  providers.some((provider) => provider.provider === model.provider && provider.models.includes(model.model));

const setSessionModelRemote = async (
  sessionId: string,
  sessionHeader: string,
  provider: string,
  model: string,
): Promise<SelectedModel | null> => {
  const response = await fetch('/ui/api/session-model', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      [sessionHeader]: sessionId,
    },
    body: JSON.stringify({ provider, model }),
  });
  const payload: unknown = await response.json();
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, 'Failed to set model.'));
  }
  const parsed = parseSelectedModel((payload as { selected_model?: unknown }).selected_model);
  return parsed || { provider, model };
};

const setSessionModeRemote = async (
  sessionId: string,
  sessionHeader: string,
  mode: SessionMode,
): Promise<WorkflowSnapshot> => {
  const response = await fetch('/ui/api/mode', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      [sessionHeader]: sessionId,
    },
    body: JSON.stringify({
      mode,
      confirm: mode === 'act',
    }),
  });
  const payload: unknown = await response.json();
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, 'Failed to set mode.'));
  }
  return {
    mode: parseSessionMode((payload as { mode?: unknown }).mode),
    activePlan: parsePlanEnvelope((payload as { active_plan?: unknown }).active_plan),
    activeTask: parseTaskExecution((payload as { active_task?: unknown }).active_task),
    autoState: parseAutoState((payload as { auto_state?: unknown }).auto_state),
    modeTransitions: parseModeTransitions(
      (payload as { mode_transitions?: unknown }).mode_transitions,
    ),
  };
};

const draftPlanRemote = async (
  sessionId: string,
  sessionHeader: string,
  goal: string,
): Promise<WorkflowSnapshot> => {
  const response = await fetch('/ui/api/plan/draft', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      [sessionHeader]: sessionId,
    },
    body: JSON.stringify({ goal }),
  });
  const payload: unknown = await response.json();
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, 'Failed to draft plan.'));
  }
  return {
    mode: parseSessionMode((payload as { mode?: unknown }).mode),
    activePlan: parsePlanEnvelope((payload as { active_plan?: unknown }).active_plan),
    activeTask: parseTaskExecution((payload as { active_task?: unknown }).active_task),
    autoState: parseAutoState((payload as { auto_state?: unknown }).auto_state),
    modeTransitions: parseModeTransitions(
      (payload as { mode_transitions?: unknown }).mode_transitions,
    ),
  };
};

const approvePlanRemote = async (
  sessionId: string,
  sessionHeader: string,
): Promise<WorkflowSnapshot> => {
  const response = await fetch('/ui/api/plan/approve', {
    method: 'POST',
    headers: {
      [sessionHeader]: sessionId,
    },
  });
  const payload: unknown = await response.json();
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, 'Failed to approve plan.'));
  }
  return {
    mode: parseSessionMode((payload as { mode?: unknown }).mode),
    activePlan: parsePlanEnvelope((payload as { active_plan?: unknown }).active_plan),
    activeTask: parseTaskExecution((payload as { active_task?: unknown }).active_task),
    autoState: parseAutoState((payload as { auto_state?: unknown }).auto_state),
    modeTransitions: parseModeTransitions(
      (payload as { mode_transitions?: unknown }).mode_transitions,
    ),
  };
};

const executePlanRemote = async (
  sessionId: string,
  sessionHeader: string,
): Promise<WorkflowSnapshot & { decision: UiDecision | null }> => {
  const response = await fetch('/ui/api/plan/execute', {
    method: 'POST',
    headers: {
      [sessionHeader]: sessionId,
    },
    body: JSON.stringify({}),
  });
  const payload: unknown = await response.json();
  if (!response.ok) {
    const error = (payload as { error?: { code?: unknown } }).error;
    const code = error && typeof error === 'object' ? (error as { code?: unknown }).code : null;
    if (code !== 'switch_to_act_required') {
      throw new Error(extractErrorMessage(payload, 'Failed to execute plan.'));
    }
  }
  return {
    mode: parseSessionMode((payload as { mode?: unknown }).mode),
    activePlan: parsePlanEnvelope((payload as { active_plan?: unknown }).active_plan),
    activeTask: parseTaskExecution((payload as { active_task?: unknown }).active_task),
    autoState: parseAutoState((payload as { auto_state?: unknown }).auto_state),
    modeTransitions: parseModeTransitions(
      (payload as { mode_transitions?: unknown }).mode_transitions,
    ),
    decision: parseUiDecision((payload as { decision?: unknown }).decision),
  };
};

const cancelPlanRemote = async (
  sessionId: string,
  sessionHeader: string,
): Promise<WorkflowSnapshot> => {
  const response = await fetch('/ui/api/plan/cancel', {
    method: 'POST',
    headers: {
      [sessionHeader]: sessionId,
    },
  });
  const payload: unknown = await response.json();
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, 'Failed to cancel plan.'));
  }
  return {
    mode: parseSessionMode((payload as { mode?: unknown }).mode),
    activePlan: parsePlanEnvelope((payload as { active_plan?: unknown }).active_plan),
    activeTask: parseTaskExecution((payload as { active_task?: unknown }).active_task),
    autoState: parseAutoState((payload as { auto_state?: unknown }).auto_state),
    modeTransitions: parseModeTransitions(
      (payload as { mode_transitions?: unknown }).mode_transitions,
    ),
  };
};

export function useSessionRuntimeController({
  sessionHeader,
  providerModels,
  loadSessions,
  loadModels,
  loadComposerSettings,
  onStatusMessage,
  transportRef,
}: UseSessionRuntimeControllerOptions): SessionRuntimeControllerResult {
  const resyncFetchInFlightRef = useRef(false);
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<SelectedModel | null>(null);
  const [savingModel, setSavingModel] = useState(false);
  const [pendingDecision, setPendingDecision] = useState<UiDecision | null>(null);
  const [decisionBusy, setDecisionBusy] = useState(false);
  const [decisionError, setDecisionError] = useState<string | null>(null);
  const [sessionSecuritySummary, setSessionSecuritySummary] = useState<SessionSecuritySummary>(
    DEFAULT_SESSION_SECURITY_SUMMARY,
  );
  const [workspaceRoot, setWorkspaceRoot] = useState('');
  const [sessionMode, setSessionMode] = useState<SessionMode>('ask');
  const [activePlan, setActivePlan] = useState<PlanEnvelope | null>(null);
  const [activeTask, setActiveTask] = useState<TaskExecutionState | null>(null);
  const [autoState, setAutoState] = useState<AutoState | null>(null);
  const [modeTransitions, setModeTransitions] = useState<ModeTransitionsContract | null>(null);
  const [modeBusy, setModeBusy] = useState(false);
  const [modeError, setModeError] = useState<string | null>(null);
  const [lastModelApplied, setLastModelApplied] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);

  const applyRuntimePayload = (payload: unknown) => {
    const body = payload as {
      decision?: unknown;
      selected_model?: unknown;
      mode?: unknown;
      active_plan?: unknown;
      active_task?: unknown;
      auto_state?: unknown;
      mode_transitions?: unknown;
    };
    if (body.decision !== undefined) {
      setPendingDecision(parseUiDecision(body.decision));
      setDecisionError(null);
    }
    const parsedModel = parseSelectedModel(body.selected_model);
    if (parsedModel) {
      setSelectedModel(parsedModel);
      saveLastModel(parsedModel);
    }
    if (body.mode !== undefined) {
      setSessionMode(parseSessionMode(body.mode));
    }
    if (body.active_plan !== undefined) {
      setActivePlan(parsePlanEnvelope(body.active_plan));
    }
    if (body.active_task !== undefined) {
      setActiveTask(parseTaskExecution(body.active_task));
    }
    if (body.auto_state !== undefined) {
      setAutoState(parseAutoState(body.auto_state));
    }
    if (body.mode_transitions !== undefined) {
      setModeTransitions(parseModeTransitions(body.mode_transitions));
    }
  };

  const refreshSessionSecuritySummary = async (sessionId?: string | null): Promise<void> => {
    const activeSessionId = sessionId ?? selectedConversation;
    if (!activeSessionId) {
      setSessionSecuritySummary(DEFAULT_SESSION_SECURITY_SUMMARY);
      return;
    }
    const summary = await loadSessionSecuritySummary(activeSessionId, sessionHeader);
    setSessionSecuritySummary(summary);
  };

  const loadConversation = async (sessionId: string): Promise<SelectedModel | null> => {
    const [sessionResponse, historyResponse, workspaceHistoryResponse, outputResponse, filesResponse] =
      await Promise.all([
        fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}`, {
          headers: {
            [sessionHeader]: sessionId,
          },
        }),
        fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}/history`, {
          headers: {
            [sessionHeader]: sessionId,
          },
        }),
        fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}/history?lane=workspace`, {
          headers: {
            [sessionHeader]: sessionId,
          },
        }),
        fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}/output`, {
          headers: {
            [sessionHeader]: sessionId,
          },
        }),
        fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}/files`, {
          headers: {
            [sessionHeader]: sessionId,
          },
        }),
      ]);

    const [sessionPayload, historyPayload, workspaceHistoryPayload, outputPayload, filesPayload]: unknown[] =
      await Promise.all([
        sessionResponse.json(),
        historyResponse.json(),
        workspaceHistoryResponse.json(),
        outputResponse.json(),
        filesResponse.json(),
      ]);

    if (!sessionResponse.ok) {
      throw new Error(extractErrorMessage(sessionPayload, 'Failed to load chat session.'));
    }
    if (!historyResponse.ok) {
      throw new Error(extractErrorMessage(historyPayload, 'Failed to load chat history.'));
    }
    if (!workspaceHistoryResponse.ok) {
      throw new Error(
        extractErrorMessage(workspaceHistoryPayload, 'Failed to load workspace history.'),
      );
    }
    if (!outputResponse.ok) {
      throw new Error(extractErrorMessage(outputPayload, 'Failed to load canvas output.'));
    }
    if (!filesResponse.ok) {
      throw new Error(extractErrorMessage(filesPayload, 'Failed to load session files.'));
    }

    const session = (sessionPayload as { session?: { selected_model?: unknown } }).session;
    transportRef.current?.applyLoadedConversation({
      chatMessages: parseMessages((historyPayload as { messages?: unknown }).messages),
      workspaceMessages: parseMessages(
        (workspaceHistoryPayload as { messages?: unknown }).messages,
      ),
      outputContent: parseSessionOutput((outputPayload as { output?: unknown }).output).content,
      files: parseSessionFiles((filesPayload as { files?: unknown }).files),
      artifacts: parseSessionArtifacts((session as { artifacts?: unknown } | undefined)?.artifacts),
    });
    setPendingDecision(parseUiDecision((session as { decision?: unknown } | undefined)?.decision));
    setDecisionError(null);
    try {
      await refreshSessionSecuritySummary(sessionId);
    } catch {
      setSessionSecuritySummary(DEFAULT_SESSION_SECURITY_SUMMARY);
    }
    setWorkspaceRoot(
      parseWorkspaceRoot((session as { workspace_root?: unknown } | undefined)?.workspace_root),
    );
    setSessionMode(parseSessionMode((session as { mode?: unknown } | undefined)?.mode));
    setActivePlan(parsePlanEnvelope((session as { active_plan?: unknown } | undefined)?.active_plan));
    setActiveTask(parseTaskExecution((session as { active_task?: unknown } | undefined)?.active_task));
    setAutoState(parseAutoState((session as { auto_state?: unknown } | undefined)?.auto_state));
    setModeTransitions(
      parseModeTransitions((session as { mode_transitions?: unknown } | undefined)?.mode_transitions),
    );
    const parsedSelectedModel = parseSelectedModel(session?.selected_model);
    setSelectedModel(parsedSelectedModel);
    return parsedSelectedModel;
  };

  const createConversation = async (): Promise<{
    sessionId: string | null;
    selectedModel: SelectedModel | null;
  }> => {
    const response = await fetch('/ui/api/sessions', {
      method: 'POST',
    });
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to create chat.'));
    }
    const headerSession = response.headers.get(sessionHeader);
    const payloadSession = extractSessionIdFromPayload(payload);
    const nextSession = (headerSession && headerSession.trim()) || payloadSession || null;
    const session = (
      payload as {
        session?: { messages?: unknown; workspace_messages?: unknown; selected_model?: unknown };
      }
    ).session;
    transportRef.current?.applyLoadedConversation({
      chatMessages: parseMessages(session?.messages),
      workspaceMessages: parseMessages(
        (session as { workspace_messages?: unknown } | undefined)?.workspace_messages,
      ),
      outputContent: parseSessionOutput((session as { output?: unknown } | undefined)?.output).content,
      files: parseSessionFiles((session as { files?: unknown } | undefined)?.files),
      artifacts: parseSessionArtifacts((session as { artifacts?: unknown } | undefined)?.artifacts),
    });
    const sessionModel = parseSelectedModel(session?.selected_model);
    setPendingDecision(parseUiDecision((session as { decision?: unknown } | undefined)?.decision));
    setDecisionError(null);
    setWorkspaceRoot(
      parseWorkspaceRoot((session as { workspace_root?: unknown } | undefined)?.workspace_root),
    );
    setSessionMode(parseSessionMode((session as { mode?: unknown } | undefined)?.mode));
    setActivePlan(parsePlanEnvelope((session as { active_plan?: unknown } | undefined)?.active_plan));
    setActiveTask(parseTaskExecution((session as { active_task?: unknown } | undefined)?.active_task));
    setAutoState(parseAutoState((session as { auto_state?: unknown } | undefined)?.auto_state));
    setModeTransitions(
      parseModeTransitions((session as { mode_transitions?: unknown } | undefined)?.mode_transitions),
    );
    setSelectedModel(sessionModel);
    if (nextSession) {
      try {
        await refreshSessionSecuritySummary(nextSession);
      } catch {
        setSessionSecuritySummary(DEFAULT_SESSION_SECURITY_SUMMARY);
      }
    } else {
      setSessionSecuritySummary(DEFAULT_SESSION_SECURITY_SUMMARY);
    }
    return { sessionId: nextSession, selectedModel: sessionModel };
  };

  useEffect(() => {
    let cancelled = false;

    const bootstrap = async () => {
      try {
        const modelsPromise = loadModels();
        const composerPromise = loadComposerSettings().catch(() => null);
        const listedSessions = await loadSessions();

        const storedSession = loadLastSessionId();
        const storedExists =
          storedSession && listedSessions.some((session) => session.session_id === storedSession);
        let nextSession = storedExists ? storedSession : null;
        if (!nextSession && listedSessions.length > 0) {
          nextSession = listedSessions[0].session_id;
        }
        if (!nextSession) {
          const created = await createConversation();
          nextSession = created.sessionId;
        }

        if (!cancelled) {
          let selectedFromSession: SelectedModel | null = null;
          if (nextSession) {
            setSelectedConversation(nextSession);
            selectedFromSession = await loadConversation(nextSession);
            saveLastSessionId(nextSession);
          }
          const models = await modelsPromise;
          await composerPromise;
          if (nextSession && !selectedFromSession && !lastModelApplied) {
            const lastModel = loadLastModel();
            if (lastModel && isModelAvailable(lastModel, models)) {
              try {
                const applied = await setSessionModelRemote(
                  nextSession,
                  sessionHeader,
                  lastModel.provider,
                  lastModel.model,
                );
                setSelectedModel(applied);
                saveLastModel(applied);
                setLastModelApplied(true);
              } catch (error) {
                const message =
                  error instanceof Error ? error.message : 'Failed to restore last model.';
                onStatusMessage(message);
              }
            }
          }
          onStatusMessage(null);
        }
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : 'Failed to initialize chat.';
          onStatusMessage(message);
        }
      }
    };

    void bootstrap();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedConversation) {
      setSessionSecuritySummary(DEFAULT_SESSION_SECURITY_SUMMARY);
      setWorkspaceRoot('');
      setModeTransitions(null);
    }
  }, [selectedConversation]);

  const handleSelectConversation = async (sessionId: string) => {
    if (!sessionId || sessionId === selectedConversation) {
      return;
    }
    setSelectedConversation(sessionId);
    transportRef.current?.clearConversationState();
    setPendingDecision(null);
    setDecisionError(null);
    try {
      await loadConversation(sessionId);
      saveLastSessionId(sessionId);
      onStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load selected chat.';
      onStatusMessage(message);
    }
  };

  const handleCreateConversation = async () => {
    try {
      const created = await createConversation();
      const nextSession = created.sessionId;
      if (!nextSession) {
        onStatusMessage('Failed to create chat.');
        return;
      }
      setSelectedConversation(nextSession);
      setPendingDecision(null);
      setDecisionError(null);
      await loadConversation(nextSession);
      saveLastSessionId(nextSession);
      await loadSessions();
      if (!created.selectedModel && providerModels.length > 0) {
        const lastModel = loadLastModel();
        if (lastModel && isModelAvailable(lastModel, providerModels)) {
          try {
            const applied = await setSessionModelRemote(
              nextSession,
              sessionHeader,
              lastModel.provider,
              lastModel.model,
            );
            setSelectedModel(applied);
            saveLastModel(applied);
            setLastModelApplied(true);
          } catch (error) {
            const message =
              error instanceof Error ? error.message : 'Failed to restore last model.';
            onStatusMessage(message);
          }
        }
      }
      onStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create chat.';
      onStatusMessage(message);
    }
  };

  const handleDeleteConversation = async (sessionId: string) => {
    if (!sessionId || deletingSessionId === sessionId) {
      return;
    }
    setDeletingSessionId(sessionId);
    try {
      const response = await fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: 'DELETE',
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to delete chat.'));
      }

      const updatedSessions = await loadSessions();

      if (selectedConversation === sessionId) {
        if (updatedSessions.length > 0) {
          const nextSession = updatedSessions[0].session_id;
          setSelectedConversation(nextSession);
          await loadConversation(nextSession);
          saveLastSessionId(nextSession);
        } else {
          const created = await createConversation();
          if (created.sessionId) {
            setSelectedConversation(created.sessionId);
            await loadConversation(created.sessionId);
            await loadSessions();
            saveLastSessionId(created.sessionId);
          } else {
            setSelectedConversation(null);
            transportRef.current?.clearConversationState();
            setSelectedModel(null);
            saveLastSessionId(null);
          }
        }
      }

      onStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to delete chat.';
      onStatusMessage(message);
    } finally {
      setDeletingSessionId(null);
    }
  };

  const handleSetModel = async (provider: string, model: string): Promise<boolean> => {
    if (!selectedConversation || savingModel) {
      return false;
    }
    setSavingModel(true);
    try {
      const nextModel = await setSessionModelRemote(selectedConversation, sessionHeader, provider, model);
      setSelectedModel(nextModel);
      saveLastModel(nextModel);
      onStatusMessage(null);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to set model.';
      onStatusMessage(message);
      return false;
    } finally {
      setSavingModel(false);
    }
  };

  const applyWorkflowSnapshot = (snapshot: WorkflowSnapshot) => {
    setSessionMode(snapshot.mode);
    setActivePlan(snapshot.activePlan);
    setActiveTask(snapshot.activeTask);
    setAutoState(snapshot.autoState);
    setModeTransitions(snapshot.modeTransitions);
  };

  const handleChangeMode = async (mode: SessionMode): Promise<void> => {
    if (!selectedConversation || modeBusy) {
      return;
    }
    setModeBusy(true);
    setModeError(null);
    try {
      applyWorkflowSnapshot(await setSessionModeRemote(selectedConversation, sessionHeader, mode));
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to change mode.';
      setModeError(message);
      onStatusMessage(message);
    } finally {
      setModeBusy(false);
    }
  };

  const handlePlanDraft = async (goal: string): Promise<void> => {
    if (!selectedConversation || modeBusy) {
      return;
    }
    setModeBusy(true);
    setModeError(null);
    try {
      applyWorkflowSnapshot(await draftPlanRemote(selectedConversation, sessionHeader, goal));
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to draft plan.';
      setModeError(message);
      onStatusMessage(message);
    } finally {
      setModeBusy(false);
    }
  };

  const handlePlanApprove = async (): Promise<void> => {
    if (!selectedConversation || modeBusy) {
      return;
    }
    setModeBusy(true);
    setModeError(null);
    try {
      applyWorkflowSnapshot(await approvePlanRemote(selectedConversation, sessionHeader));
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to approve plan.';
      setModeError(message);
      onStatusMessage(message);
    } finally {
      setModeBusy(false);
    }
  };

  const handlePlanExecute = async (): Promise<void> => {
    if (!selectedConversation || modeBusy) {
      return;
    }
    setModeBusy(true);
    setModeError(null);
    try {
      const updated = await executePlanRemote(selectedConversation, sessionHeader);
      applyWorkflowSnapshot(updated);
      setPendingDecision(updated.decision);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to execute plan.';
      setModeError(message);
      onStatusMessage(message);
    } finally {
      setModeBusy(false);
    }
  };

  const handlePlanCancel = async (): Promise<void> => {
    if (!selectedConversation || modeBusy) {
      return;
    }
    setModeBusy(true);
    setModeError(null);
    try {
      applyWorkflowSnapshot(await cancelPlanRemote(selectedConversation, sessionHeader));
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to cancel plan.';
      setModeError(message);
      onStatusMessage(message);
    } finally {
      setModeBusy(false);
    }
  };

  const handleDecisionRespond = async (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
    onResume?: (resume: unknown) => void,
  ) => {
    if (!selectedConversation || !pendingDecision) {
      return;
    }
    setDecisionBusy(true);
    setDecisionError(null);
    try {
      const response = await fetch('/ui/api/decision/respond', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [sessionHeader]: selectedConversation,
        },
        body: JSON.stringify({
          session_id: selectedConversation,
          decision_id: pendingDecision.id,
          choice,
          edited_action: choice === 'edit_and_approve' ? (editedPayload ?? {}) : null,
          edited_plan: choice === 'edit_plan' ? (editedPayload ?? {}) : null,
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to resolve decision.'));
      }
      transportRef.current?.applySessionPayload(payload, { applyDisplay: false });
      applyRuntimePayload(payload);
      const resumedWorkspaceRoot = parseDecisionResumeWorkspaceRoot(
        (payload as { resume?: unknown }).resume,
      );
      if (resumedWorkspaceRoot) {
        setWorkspaceRoot(resumedWorkspaceRoot);
      }
      onResume?.((payload as { resume?: unknown }).resume);
      await loadSessions();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to resolve decision.';
      setDecisionError(message);
    } finally {
      setDecisionBusy(false);
    }
  };

  useEffect(() => {
    if (!selectedConversation) {
      return;
    }
    let cancelled = false;
    const refreshWorkflowSnapshot = async () => {
      if (resyncFetchInFlightRef.current) {
        return;
      }
      resyncFetchInFlightRef.current = true;
      try {
        const stateResponse = await fetch('/ui/api/state', {
          headers: {
            [sessionHeader]: selectedConversation,
          },
        });
        const statePayload: unknown = await stateResponse.json();
        if (!stateResponse.ok || cancelled) {
          return;
        }
        const snapshot = statePayload as {
          mode?: unknown;
          active_plan?: unknown;
          active_task?: unknown;
          auto_state?: unknown;
          pending_decision?: unknown;
        };
        setSessionMode(parseSessionMode(snapshot.mode));
        setActivePlan(parsePlanEnvelope(snapshot.active_plan));
        setActiveTask(parseTaskExecution(snapshot.active_task));
        setAutoState(parseAutoState(snapshot.auto_state));
        setPendingDecision(parseUiDecision(snapshot.pending_decision));
        setDecisionError(null);
      } catch {
        // Keep live session state; next event or manual action will recover.
      } finally {
        resyncFetchInFlightRef.current = false;
      }
    };
    void refreshWorkflowSnapshot();
    return () => {
      cancelled = true;
      resyncFetchInFlightRef.current = false;
    };
  }, [selectedConversation]);

  return {
    selectedConversation,
    selectedModel,
    savingModel,
    sessionMode,
    activePlan,
    activeTask,
    autoState,
    modeTransitions,
    pendingDecision,
    decisionBusy,
    decisionError,
    sessionSecuritySummary,
    workspaceRoot,
    modeBusy,
    modeError,
    handleSelectConversation,
    handleCreateConversation,
    handleDeleteConversation,
    handleSetModel,
    handleChangeMode,
    handlePlanDraft,
    handlePlanApprove,
    handlePlanExecute,
    handlePlanCancel,
    handleDecisionRespond,
    applyWorkspaceRoot: (nextWorkspaceRoot: string) => {
      setWorkspaceRoot(parseWorkspaceRoot(nextWorkspaceRoot));
    },
    refreshSessionSecuritySummary,
    applyRuntimePayload,
  };
}
