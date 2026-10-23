import { useEffect, useRef, useState, type MutableRefObject } from 'react';

import { RootReconcileGuard } from './root-reconcile';
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
  ComputerActivityEvent,
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

type WorkflowField = keyof WorkflowSnapshot | 'decision';
type WorkflowEvidence = { version: number; signature: string; planId?: string; planRevision?: number };

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
  computerEvents: ComputerActivityEvent[];
  modeBusy: boolean;
  planCancelling: boolean;
  modeError: string | null;
  handleSelectConversation: (sessionId: string) => Promise<void>;
  handleCreateConversation: () => Promise<void>;
  handleDeleteConversation: (sessionId: string) => Promise<void>;
  handleSetModel: (provider: string, model: string) => Promise<boolean>;
  handleChangeMode: (mode: SessionMode) => Promise<void>;
  handlePlanDraft: (goal: string) => Promise<void>;
  handlePlanApprove: (plan: PlanEnvelope) => Promise<void>;
  handlePlanExecute: (plan: PlanEnvelope) => Promise<void>;
  handlePlanCancel: () => Promise<void>;
  handleDecisionRespond: (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
    onResume?: (resume: unknown) => void,
    onRejected?: () => void,
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
  plan: PlanEnvelope,
): Promise<WorkflowSnapshot> => {
  const response = await fetch('/ui/api/plan/approve', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      [sessionHeader]: sessionId,
    },
    body: JSON.stringify({ plan_id: plan.plan_id, plan_hash: plan.plan_hash, plan_revision: plan.plan_revision }),
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
  plan: PlanEnvelope,
): Promise<WorkflowSnapshot & { decision: UiDecision | null }> => {
  const response = await fetch('/ui/api/plan/execute', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Idempotency-Key': crypto.randomUUID(),
      [sessionHeader]: sessionId,
    },
    body: JSON.stringify({ plan_id: plan.plan_id, plan_hash: plan.plan_hash, plan_revision: plan.plan_revision }),
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
  const reconcileGuardRef = useRef(new RootReconcileGuard());
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<SelectedModel | null>(null);
  const [savingModel, setSavingModel] = useState(false);
  const modelRequestRef = useRef<symbol | null>(null);
  const workflowRequestRef = useRef<symbol | null>(null);
  const decisionRequestRef = useRef<symbol | null>(null);
  const cancelRequestRef = useRef<symbol | null>(null);
  const workflowVersionRef = useRef(0);
  const workflowEvidenceRef = useRef(new Map<WorkflowField, WorkflowEvidence>());
  const decisionIdentityRef = useRef<string | null>(null);
  const [planCancelling, setPlanCancelling] = useState(false);
  const initialStateAbortRef = useRef<AbortController | null>(null);
  const beginConversationChange = () => {
    modelRequestRef.current = null;
    workflowRequestRef.current = null;
    decisionRequestRef.current = null;
    cancelRequestRef.current = null;
    setDecisionBusy(false);
    setDecisionError(null);
    setPlanCancelling(false);
    initialStateAbortRef.current?.abort();
    setModeTransitions(null);
    setModeBusy(false);
    setModeError(null);
    setSavingModel(false);
    return reconcileGuardRef.current.begin();
  };
  const [pendingDecision, setPendingDecision] = useState<UiDecision | null>(null);
  const [decisionBusy, setDecisionBusy] = useState(false);
  const [decisionError, setDecisionError] = useState<string | null>(null);
  const applyDecision = (decision: UiDecision | null) => {
    if (decisionIdentityRef.current !== (decision?.id ?? null)) {
      decisionRequestRef.current = null;
      setDecisionBusy(false);
    }
    decisionIdentityRef.current = decision?.id ?? null;
    setPendingDecision(decision);
  };
  const [sessionSecuritySummary, setSessionSecuritySummary] = useState<SessionSecuritySummary>(
    DEFAULT_SESSION_SECURITY_SUMMARY,
  );
  const [workspaceRoot, setWorkspaceRoot] = useState('');
  const [sessionMode, setSessionMode] = useState<SessionMode>('ask');
  const [activePlan, setActivePlan] = useState<PlanEnvelope | null>(null);
  const [activeTask, setActiveTask] = useState<TaskExecutionState | null>(null);
  const [autoState, setAutoState] = useState<AutoState | null>(null);
  const [modeTransitions, setModeTransitions] = useState<ModeTransitionsContract | null>(null);
  const [computerEvents, setComputerEvents] = useState<ComputerActivityEvent[]>([]);
  const [modeBusy, setModeBusy] = useState(false);
  const [modeError, setModeError] = useState<string | null>(null);
  const [lastModelApplied, setLastModelApplied] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);

  const parseComputerEvents = (value: unknown): ComputerActivityEvent[] => {
    if (!Array.isArray(value)) {
      return [];
    }
    return value
      .filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
      .map((item) => ({
        kind: typeof item.kind === 'string' ? item.kind : 'tool_started',
        tool: typeof item.tool === 'string' ? item.tool : '',
        ts: typeof item.ts === 'number' ? item.ts : 0,
        ok: typeof item.ok === 'boolean' ? item.ok : undefined,
        path: typeof item.path === 'string' ? item.path : undefined,
        command: typeof item.command === 'string' ? item.command : undefined,
        error: typeof item.error === 'string' ? item.error : undefined,
        duration_ms: typeof item.duration_ms === 'number' ? item.duration_ms : undefined,
      }));
  };

  const observeWorkflow = (field: WorkflowField, value: unknown) => {
    const plan = field === 'activePlan' ? parsePlanEnvelope(value) : null;
    workflowEvidenceRef.current.set(field, { version: workflowVersionRef.current,
      signature: JSON.stringify(value), ...(plan ? { planId: plan.plan_id, planRevision: plan.plan_revision } : {}),
    });
  };
  const canApplyWorkflow = (snapshot: Partial<WorkflowSnapshot & { decision: UiDecision | null }>, version: number) => {
    const decision = workflowEvidenceRef.current.get('decision');
    if ('decision' in snapshot && decision && decision.version > version
        && decision.signature !== JSON.stringify(snapshot.decision)) return false;
    const plan = workflowEvidenceRef.current.get('activePlan');
    if (snapshot.activePlan && plan && plan.version > version
        && plan.planId === snapshot.activePlan.plan_id
        && plan.planRevision !== undefined && snapshot.activePlan.plan_revision > plan.planRevision) return true;
    return [...workflowEvidenceRef.current].every(([field, evidence]) => evidence.version <= version
      || !(field in snapshot) || evidence.signature === JSON.stringify(snapshot[field]));
  };

  const applyRuntimePayload = (payload: unknown) => {
    const body = payload as {
      decision?: unknown;
      selected_model?: unknown;
      mode?: unknown;
      active_plan?: unknown;
      active_task?: unknown;
      auto_state?: unknown;
      mode_transitions?: unknown;
      computer_events?: unknown;
    };
    if (body.mode !== undefined || body.active_plan !== undefined || body.active_task !== undefined
        || body.auto_state !== undefined || body.decision !== undefined || body.mode_transitions !== undefined) {
      initialStateAbortRef.current?.abort();
      workflowVersionRef.current += 1;
    }
    if (body.decision !== undefined) {
      observeWorkflow('decision', parseUiDecision(body.decision));
      applyDecision(parseUiDecision(body.decision));
      setDecisionError(null);
    }
    const parsedModel = parseSelectedModel(body.selected_model);
    if (parsedModel) {
      setSelectedModel(parsedModel);
      saveLastModel(parsedModel);
    }
    if (body.mode !== undefined) {
      observeWorkflow('mode', parseSessionMode(body.mode));
      setSessionMode(parseSessionMode(body.mode));
    }
    if (body.active_plan !== undefined) {
      observeWorkflow('activePlan', parsePlanEnvelope(body.active_plan));
      setActivePlan(parsePlanEnvelope(body.active_plan));
    }
    if (body.active_task !== undefined) {
      observeWorkflow('activeTask', parseTaskExecution(body.active_task));
      setActiveTask(parseTaskExecution(body.active_task));
    }
    if (body.auto_state !== undefined) {
      observeWorkflow('autoState', parseAutoState(body.auto_state));
      setAutoState(parseAutoState(body.auto_state));
    }
    if (body.mode_transitions !== undefined) {
      observeWorkflow('modeTransitions', parseModeTransitions(body.mode_transitions));
      setModeTransitions(parseModeTransitions(body.mode_transitions));
    }
    if (body.computer_events !== undefined) {
      setComputerEvents(parseComputerEvents(body.computer_events));
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

  const loadConversation = async (
    sessionId: string,
    generation: number,
  ): Promise<SelectedModel | null> => {
    const [sessionResponse, outputResponse, filesResponse] =
      await Promise.all([
        fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}`, {
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

    const [sessionPayload, outputPayload, filesPayload]: unknown[] =
      await Promise.all([
        sessionResponse.json(),
        outputResponse.json(),
        filesResponse.json(),
      ]);

    if (!sessionResponse.ok) {
      throw new Error(extractErrorMessage(sessionPayload, 'Failed to load chat session.'));
    }
    if (!outputResponse.ok) {
      throw new Error(extractErrorMessage(outputPayload, 'Failed to load canvas output.'));
    }
    if (!filesResponse.ok) {
      throw new Error(extractErrorMessage(filesPayload, 'Failed to load session files.'));
    }

    const session = (sessionPayload as { session?: { selected_model?: unknown } }).session;
    let securitySummary = DEFAULT_SESSION_SECURITY_SUMMARY;
    try {
      securitySummary = await loadSessionSecuritySummary(sessionId, sessionHeader);
    } catch {
      securitySummary = DEFAULT_SESSION_SECURITY_SUMMARY;
    }
    if (!reconcileGuardRef.current.isCurrent(generation)) {
      return null;
    }
    transportRef.current?.applyLoadedConversation({
      chatMessages: parseMessages((session as { messages?: unknown } | undefined)?.messages),
      outputContent: parseSessionOutput((outputPayload as { output?: unknown }).output).content,
      files: parseSessionFiles((filesPayload as { files?: unknown }).files),
      artifacts: parseSessionArtifacts((session as { artifacts?: unknown } | undefined)?.artifacts),
    });
    applyDecision(parseUiDecision((session as { decision?: unknown } | undefined)?.decision));
    setDecisionError(null);
    setSessionSecuritySummary(securitySummary);
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
    setComputerEvents(
      parseComputerEvents((session as { computer_events?: unknown } | undefined)?.computer_events),
    );
    const parsedSelectedModel = parseSelectedModel(session?.selected_model);
    setSelectedModel(parsedSelectedModel);
    return parsedSelectedModel;
  };

  const createConversation = async (generation: number, activateOnSuccess = false): Promise<{
    generation: number;
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
    const session = (payload as { session?: { messages?: unknown; selected_model?: unknown } }).session;
    let securitySummary = DEFAULT_SESSION_SECURITY_SUMMARY;
    if (nextSession) {
      try {
        securitySummary = await loadSessionSecuritySummary(nextSession, sessionHeader);
      } catch {
        securitySummary = DEFAULT_SESSION_SECURITY_SUMMARY;
      }
    }
    if (!nextSession || !reconcileGuardRef.current.isCurrent(generation)) {
      return {
        generation,
        sessionId: nextSession,
        selectedModel: parseSelectedModel(session?.selected_model),
      };
    }
    if (activateOnSuccess) generation = beginConversationChange();
    transportRef.current?.applyLoadedConversation({
      chatMessages: parseMessages(session?.messages),
      outputContent: parseSessionOutput((session as { output?: unknown } | undefined)?.output).content,
      files: parseSessionFiles((session as { files?: unknown } | undefined)?.files),
      artifacts: parseSessionArtifacts((session as { artifacts?: unknown } | undefined)?.artifacts),
    });
    const sessionModel = parseSelectedModel(session?.selected_model);
    applyDecision(parseUiDecision((session as { decision?: unknown } | undefined)?.decision));
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
    setComputerEvents(
      parseComputerEvents((session as { computer_events?: unknown } | undefined)?.computer_events),
    );
    setSelectedModel(sessionModel);
    setSessionSecuritySummary(securitySummary);
    return { generation, sessionId: nextSession, selectedModel: sessionModel };
  };

  const applySessionModel = async (sessionId: string, generation: number, provider: string, model: string): Promise<boolean> => {
    if (!reconcileGuardRef.current.isCurrent(generation) || modelRequestRef.current !== null) {
      return false;
    }
    setSavingModel(true);
    const request = Symbol();
    modelRequestRef.current = request;
    const isCurrent = () => modelRequestRef.current === request
      && reconcileGuardRef.current.isCurrent(generation);
    try {
      const nextModel = await setSessionModelRemote(sessionId, sessionHeader, provider, model);
      if (!isCurrent()) return false;
      setSelectedModel(nextModel);
      saveLastModel(nextModel);
      onStatusMessage(null);
      return true;
    } catch (error) {
      if (!isCurrent()) return false;
      const message = error instanceof Error ? error.message : 'Failed to set model.';
      onStatusMessage(message);
      return false;
    } finally {
      if (isCurrent()) {
        modelRequestRef.current = null;
        setSavingModel(false);
      }
    }
  };

  useEffect(() => {
    let cancelled = false;
    const generation = beginConversationChange();

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
          const created = await createConversation(generation);
          nextSession = created.sessionId;
        }

        if (!cancelled && reconcileGuardRef.current.isCurrent(generation)) {
          let selectedFromSession: SelectedModel | null = null;
          if (nextSession) {
            setSelectedConversation(nextSession);
            selectedFromSession = await loadConversation(nextSession, generation);
            if (!reconcileGuardRef.current.isCurrent(generation)) {
              return;
            }
            saveLastSessionId(nextSession);
          }
          const models = await modelsPromise;
          await composerPromise;
          if (cancelled || !reconcileGuardRef.current.isCurrent(generation)) {
            return;
          }
          if (nextSession && !selectedFromSession && !lastModelApplied) {
            const lastModel = loadLastModel();
            if (lastModel && isModelAvailable(lastModel, models)) {
              const applied = await applySessionModel(
                nextSession, generation, lastModel.provider, lastModel.model,
              );
              if (!applied || !reconcileGuardRef.current.isCurrent(generation)) return;
              setLastModelApplied(true);
            }
          }
          onStatusMessage(null);
        }
      } catch (error) {
        if (!cancelled && reconcileGuardRef.current.isCurrent(generation)) {
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
    const generation = beginConversationChange();
    setSelectedConversation(sessionId);
    transportRef.current?.clearConversationState();
    applyDecision(null);
    setDecisionError(null);
    try {
      await loadConversation(sessionId, generation);
      if (!reconcileGuardRef.current.isCurrent(generation)) {
        return;
      }
      saveLastSessionId(sessionId);
      onStatusMessage(null);
      void _reconcileWorkspaceRoot(sessionId, generation);
    } catch (error) {
      if (!reconcileGuardRef.current.isCurrent(generation)) {
        return;
      }
      const message = error instanceof Error ? error.message : 'Failed to load selected chat.';
      onStatusMessage(message);
    }
  };

  const _reconcileWorkspaceRoot = async (
    sessionId: string,
    generation: number,
  ): Promise<void> => {
    try {
      const rootResponse = await fetch('/ui/api/workspace/root', {
        headers: { [sessionHeader]: sessionId },
      });
      if (!reconcileGuardRef.current.isCurrent(generation)) {
        return;
      }
      if (rootResponse.ok) {
        const rootPayload: unknown = await rootResponse.json();
        if (!reconcileGuardRef.current.isCurrent(generation)) {
          return;
        }
        const rootPath = parseWorkspaceRoot(
          (rootPayload as { root_path?: unknown }).root_path,
        );
        if (rootPath) {
          setWorkspaceRoot(rootPath);
        }
      }
    } catch {
      // non-critical reconciliation; root already set from session load
    }
  };

  const handleCreateConversation = async () => {
    let generation = reconcileGuardRef.current.snapshot();
    try {
      const created = await createConversation(generation, true);
      generation = created.generation;
      if (!reconcileGuardRef.current.isCurrent(generation)) {
        return;
      }
      const nextSession = created.sessionId;
      if (!nextSession) {
        onStatusMessage('Failed to create chat.');
        return;
      }
      setSelectedConversation(nextSession);
      applyDecision(null);
      setDecisionError(null);
      await loadConversation(nextSession, generation);
      if (!reconcileGuardRef.current.isCurrent(generation)) {
        return;
      }
      saveLastSessionId(nextSession);
      void _reconcileWorkspaceRoot(nextSession, generation);
      await loadSessions();
      if (!reconcileGuardRef.current.isCurrent(generation)) {
        return;
      }
      if (!created.selectedModel && providerModels.length > 0) {
        const lastModel = loadLastModel();
        if (lastModel && isModelAvailable(lastModel, providerModels)) {
          const applied = await applySessionModel(
            nextSession, generation, lastModel.provider, lastModel.model,
          );
          if (!applied || !reconcileGuardRef.current.isCurrent(generation)) return;
          setLastModelApplied(true);
        }
      }
      onStatusMessage(null);
    } catch (error) {
      if (!reconcileGuardRef.current.isCurrent(generation)) {
        return;
      }
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
          const generation = beginConversationChange();
          setSelectedConversation(nextSession);
          await loadConversation(nextSession, generation);
          if (!reconcileGuardRef.current.isCurrent(generation)) {
            return;
          }
          saveLastSessionId(nextSession);
        } else {
          const generation = beginConversationChange();
          const created = await createConversation(generation);
          if (!reconcileGuardRef.current.isCurrent(generation)) {
            return;
          }
          if (created.sessionId) {
            setSelectedConversation(created.sessionId);
            await loadConversation(created.sessionId, generation);
            if (!reconcileGuardRef.current.isCurrent(generation)) {
              return;
            }
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
    if (!selectedConversation || savingModel) return false;
    return applySessionModel(selectedConversation, reconcileGuardRef.current.snapshot(), provider, model);
  };

  const applyWorkflowSnapshot = (snapshot: WorkflowSnapshot) => {
    setSessionMode(snapshot.mode);
    setActivePlan(snapshot.activePlan);
    setActiveTask(snapshot.activeTask);
    setAutoState(snapshot.autoState);
    setModeTransitions(snapshot.modeTransitions);
  };

  const runWorkflowRequest = async (
    request: (sessionId: string) => Promise<WorkflowSnapshot & { decision?: UiDecision | null }>,
    fallbackError: string,
    cancellation = false,
  ): Promise<void> => {
    if (!selectedConversation || cancelRequestRef.current
        || (!cancellation && (workflowRequestRef.current || decisionRequestRef.current))) return;
    const generation = reconcileGuardRef.current.snapshot();
    initialStateAbortRef.current?.abort();
    const token = Symbol('workflow-request');
    const version = workflowVersionRef.current;
    if (cancellation) {
      cancelRequestRef.current = token;
      decisionRequestRef.current = null;
      setDecisionBusy(false);
      setDecisionError(null);
      setPlanCancelling(true);
    }
    workflowRequestRef.current = token;
    const current = () => workflowRequestRef.current === token
      && reconcileGuardRef.current.isCurrent(generation);
    setModeBusy(true);
    setModeError(null);
    try {
      const updated = await request(selectedConversation);
      if (!current()) return;
      if (canApplyWorkflow(updated, version)) {
        applyWorkflowSnapshot(updated);
        if ('decision' in updated) applyDecision(updated.decision ?? null);
      }
    } catch (error) {
      if (!current()) return;
      const message = error instanceof Error ? error.message : fallbackError;
      setModeError(message);
      onStatusMessage(message);
    } finally {
      if (current()) {
        workflowRequestRef.current = null;
        if (cancelRequestRef.current === token) {
          cancelRequestRef.current = null;
          setPlanCancelling(false);
        }
        setModeBusy(false);
      }
    }
  };

  const handleChangeMode = (mode: SessionMode): Promise<void> => runWorkflowRequest(
    (sessionId) => setSessionModeRemote(sessionId, sessionHeader, mode), 'Failed to change mode.',
  );
  const handlePlanDraft = (goal: string): Promise<void> => runWorkflowRequest(
    (sessionId) => draftPlanRemote(sessionId, sessionHeader, goal), 'Failed to draft plan.',
  );
  const handlePlanApprove = (plan: PlanEnvelope): Promise<void> => runWorkflowRequest(
    (sessionId) => approvePlanRemote(sessionId, sessionHeader, plan), 'Failed to approve plan.',
  );
  const handlePlanExecute = (plan: PlanEnvelope): Promise<void> => runWorkflowRequest(
    (sessionId) => executePlanRemote(sessionId, sessionHeader, plan), 'Failed to execute plan.',
  );
  const handlePlanCancel = (): Promise<void> => runWorkflowRequest(
    (sessionId) => cancelPlanRemote(sessionId, sessionHeader), 'Failed to cancel plan.', true,
  );

  const handleDecisionRespond = async (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
    onResume?: (resume: unknown) => void,
    onRejected?: () => void,
  ) => {
    if (!selectedConversation || !pendingDecision || decisionRequestRef.current || workflowRequestRef.current) {
      return;
    }
    const generation = reconcileGuardRef.current.snapshot();
    const token = Symbol('decision-request');
    const version = workflowVersionRef.current;
    decisionRequestRef.current = token;
    initialStateAbortRef.current?.abort();
    const current = () => decisionRequestRef.current === token
      && reconcileGuardRef.current.isCurrent(generation);
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
          edited_action: choice === 'edit_and_approve' || choice === 'edit_and_confirm' ? (editedPayload ?? {}) : null,
          edited_plan: choice === 'edit_plan' ? (editedPayload ?? {}) : null,
        }),
      });
      const payload: unknown = await response.json();
      if (!current()) return;
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to resolve decision.'));
      }
      const body = payload as Record<string, unknown>;
      const snapshot: Partial<WorkflowSnapshot & { decision: UiDecision | null }> = {};
      if ('mode' in body) snapshot.mode = parseSessionMode(body.mode);
      if ('active_plan' in body) snapshot.activePlan = parsePlanEnvelope(body.active_plan);
      if ('active_task' in body) snapshot.activeTask = parseTaskExecution(body.active_task);
      if ('auto_state' in body) snapshot.autoState = parseAutoState(body.auto_state);
      if ('mode_transitions' in body) snapshot.modeTransitions = parseModeTransitions(body.mode_transitions);
      if ('decision' in body) snapshot.decision = parseUiDecision(body.decision);
      if (canApplyWorkflow(snapshot, version)) {
        transportRef.current?.applySessionPayload(payload, { applyDisplay: false });
        applyRuntimePayload(payload);
        const resumedWorkspaceRoot = parseDecisionResumeWorkspaceRoot(
          (payload as { resume?: unknown }).resume,
        );
        if (resumedWorkspaceRoot) setWorkspaceRoot(resumedWorkspaceRoot);
      }
      if (choice === 'reject') {
        onRejected?.();
      }
      onResume?.((payload as { resume?: unknown }).resume);
      await loadSessions();
    } catch (error) {
      if (!current()) return;
      const message = error instanceof Error ? error.message : 'Failed to resolve decision.';
      setDecisionError(message);
    } finally {
      if (current()) {
        decisionRequestRef.current = null;
        setDecisionBusy(false);
      }
    }
  };

  useEffect(() => {
    if (!selectedConversation) {
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    initialStateAbortRef.current = controller;
    const refreshWorkflowSnapshot = async () => {
      if (resyncFetchInFlightRef.current) {
        return;
      }
      resyncFetchInFlightRef.current = true;
      try {
        const stateResponse = await fetch('/ui/api/state', {
          signal: controller.signal,
          headers: {
            [sessionHeader]: selectedConversation,
          },
        });
        const statePayload: unknown = await stateResponse.json();
        if (!stateResponse.ok || cancelled || controller.signal.aborted) {
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
        applyDecision(parseUiDecision(snapshot.pending_decision));
        setDecisionError(null);
      } catch {
        // Keep live session state; next event or manual action will recover.
      } finally {
        if (initialStateAbortRef.current === controller) resyncFetchInFlightRef.current = false;
      }
    };
    void refreshWorkflowSnapshot();
    return () => {
      cancelled = true;
      controller.abort();
      if (initialStateAbortRef.current === controller) resyncFetchInFlightRef.current = false;
    };
  }, [selectedConversation]);

  useEffect(() => {
    if (!selectedConversation || !activePlan || !['completed', 'failed', 'cancelled'].includes(activePlan.status)) return;
    const controller = new AbortController();
    const generation = reconcileGuardRef.current.snapshot();
    const version = workflowVersionRef.current;
    const planId = activePlan.plan_id;
    const planRevision = activePlan.plan_revision;
    void (async () => {
      try {
        const response = await fetch(`/ui/api/sessions/${encodeURIComponent(selectedConversation)}`, {
          signal: controller.signal, headers: { [sessionHeader]: selectedConversation },
        });
        const payload: unknown = await response.json();
        if (!response.ok || controller.signal.aborted || !reconcileGuardRef.current.isCurrent(generation)
            || workflowVersionRef.current !== version) return;
        const session = (payload as { session?: { active_plan?: unknown; mode_transitions?: unknown } }).session;
        const plan = parsePlanEnvelope(session?.active_plan);
        if (plan?.plan_id === planId && plan.plan_revision === planRevision) {
          setModeTransitions(parseModeTransitions(session?.mode_transitions));
        }
      } catch {
        // The terminal state stays visible; a later session load refreshes transition permissions.
      }
    })();
    return () => controller.abort();
  }, [selectedConversation, activePlan?.plan_id, activePlan?.plan_revision, activePlan?.status]);

  useEffect(() => {
    return () => {
      reconcileGuardRef.current.invalidate();
    };
  }, []);

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
    computerEvents,
    modeBusy,
    planCancelling,
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
