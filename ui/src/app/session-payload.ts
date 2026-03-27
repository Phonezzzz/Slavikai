import type { Artifact } from './components/artifacts-sidebar';
import type { CanvasMessage } from './components/canvas';
import { isSessionMode } from './types';
import type {
  AutoState,
  ChatAttachment,
  ChatMessage,
  FolderSummary,
  MessageLane,
  MessageRuntimeMeta,
  MwvReportUi,
  PlanEnvelope,
  PlanStepStatus,
  ProviderModels,
  SelectedModel,
  SessionMode,
  SessionSummary,
  TaskExecutionState,
  UiDecision,
} from './types';

export type ComposerUiSettings = {
  longPasteToFileEnabled: boolean;
  longPasteThresholdChars: number;
};

export type SessionArtifactRecord = {
  id: string;
  kind: 'output';
  title: string;
  content: string;
  createdAt: string | null;
  displayTarget: 'chat' | 'canvas';
  artifactKind: 'text' | 'file';
  fileName: string | null;
  fileExt: string | null;
  language: string | null;
  fileContent: string | null;
};

export type DisplayDecision = {
  target: 'chat' | 'canvas';
  artifactId: string | null;
  forced: boolean;
};

export type ChatStreamState = {
  streamId: string;
  content: string;
};

export type PendingUserMessage = {
  content: string;
  attachments: ChatAttachment[];
  lane: MessageLane;
};

export const DEFAULT_LONG_PASTE_THRESHOLD_CHARS = 12000;

export const DEFAULT_COMPOSER_SETTINGS: ComposerUiSettings = {
  longPasteToFileEnabled: true,
  longPasteThresholdChars: DEFAULT_LONG_PASTE_THRESHOLD_CHARS,
};

export const extractErrorMessage = (payload: unknown, fallback: string): string => {
  if (!payload || typeof payload !== 'object') {
    return fallback;
  }
  const body = payload as { error?: { message?: unknown } };
  if (body.error && typeof body.error.message === 'string' && body.error.message.trim()) {
    return body.error.message;
  }
  return fallback;
};

export const compactProviderError = (value: string): string => {
  const normalized = value.replace(/\s+/g, ' ').trim();
  if (normalized.length <= 64) {
    return normalized;
  }
  return `${normalized.slice(0, 61)}...`;
};

export const parseChatAttachments = (value: unknown): ChatAttachment[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: ChatAttachment[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as { name?: unknown; mime?: unknown; content?: unknown };
    if (
      typeof candidate.name !== 'string'
      || !candidate.name.trim()
      || typeof candidate.mime !== 'string'
      || !candidate.mime.trim()
      || typeof candidate.content !== 'string'
    ) {
      continue;
    }
    normalized.push({
      name: candidate.name.trim(),
      mime: candidate.mime.trim(),
      content: candidate.content,
    });
  }
  return normalized;
};

const isChatMessage = (value: unknown): value is ChatMessage => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const candidate = value as {
    message_id?: unknown;
    role?: unknown;
    lane?: unknown;
    content?: unknown;
    created_at?: unknown;
    trace_id?: unknown;
    parent_user_message_id?: unknown;
    attachments?: unknown;
  };
  if (typeof candidate.message_id !== 'string' || !candidate.message_id.trim()) {
    return false;
  }
  if (candidate.role !== 'user' && candidate.role !== 'assistant' && candidate.role !== 'system') {
    return false;
  }
  if (candidate.lane !== undefined && candidate.lane !== 'chat' && candidate.lane !== 'workspace') {
    return false;
  }
  if (typeof candidate.content !== 'string') {
    return false;
  }
  if (typeof candidate.created_at !== 'string' || !candidate.created_at.trim()) {
    return false;
  }
  if (candidate.trace_id !== null && typeof candidate.trace_id !== 'string') {
    return false;
  }
  if (
    candidate.parent_user_message_id !== null
    && typeof candidate.parent_user_message_id !== 'string'
  ) {
    return false;
  }
  if (candidate.attachments !== undefined && !Array.isArray(candidate.attachments)) {
    return false;
  }
  return true;
};

export const parseMessages = (value: unknown): ChatMessage[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(isChatMessage).map((message) => ({
    ...message,
    lane: (message as { lane?: unknown }).lane === 'workspace' ? 'workspace' : 'chat',
    attachments: parseChatAttachments((message as { attachments?: unknown }).attachments),
  }));
};

export const parseComposerSettings = (value: unknown): ComposerUiSettings => {
  if (!value || typeof value !== 'object') {
    return DEFAULT_COMPOSER_SETTINGS;
  }
  const settings = (value as { settings?: unknown }).settings;
  if (!settings || typeof settings !== 'object') {
    return DEFAULT_COMPOSER_SETTINGS;
  }
  const composer = (settings as { composer?: unknown }).composer;
  if (!composer || typeof composer !== 'object') {
    return DEFAULT_COMPOSER_SETTINGS;
  }
  const candidate = composer as {
    long_paste_to_file_enabled?: unknown;
    long_paste_threshold_chars?: unknown;
  };
  const enabled =
    typeof candidate.long_paste_to_file_enabled === 'boolean'
      ? candidate.long_paste_to_file_enabled
      : DEFAULT_COMPOSER_SETTINGS.longPasteToFileEnabled;
  const threshold =
    typeof candidate.long_paste_threshold_chars === 'number'
      && Number.isFinite(candidate.long_paste_threshold_chars)
      && candidate.long_paste_threshold_chars > 0
      ? Math.floor(candidate.long_paste_threshold_chars)
      : DEFAULT_COMPOSER_SETTINGS.longPasteThresholdChars;
  return {
    longPasteToFileEnabled: enabled,
    longPasteThresholdChars: threshold,
  };
};

export const parseSelectedModel = (value: unknown): SelectedModel | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const candidate = value as { provider?: unknown; model?: unknown };
  if (typeof candidate.provider !== 'string' || typeof candidate.model !== 'string') {
    return null;
  }
  return { provider: candidate.provider, model: candidate.model };
};

export const parseProviderModels = (value: unknown): ProviderModels[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const providers: ProviderModels[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as { provider?: unknown; models?: unknown; error?: unknown };
    if (typeof candidate.provider !== 'string' || !Array.isArray(candidate.models)) {
      continue;
    }
    providers.push({
      provider: candidate.provider,
      models: candidate.models.filter((entry): entry is string => typeof entry === 'string'),
      error:
        typeof candidate.error === 'string' || candidate.error === null ? candidate.error : null,
    });
  }
  return providers;
};

export const sortSessionsByUpdated = (value: SessionSummary[]): SessionSummary[] => {
  return [...value].sort((a, b) => {
    const aTime = Date.parse(a.updated_at);
    const bTime = Date.parse(b.updated_at);
    const aValue = Number.isNaN(aTime) ? 0 : aTime;
    const bValue = Number.isNaN(bTime) ? 0 : bTime;
    return bValue - aValue;
  });
};

export const parseSessions = (value: unknown): SessionSummary[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const sessions: SessionSummary[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as {
      session_id?: unknown;
      title?: unknown;
      created_at?: unknown;
      updated_at?: unknown;
      message_count?: unknown;
      chat_message_count?: unknown;
      workspace_message_count?: unknown;
      last_message_lane?: unknown;
      title_override?: unknown;
      folder_id?: unknown;
    };
    if (
      typeof candidate.session_id !== 'string'
      || typeof candidate.created_at !== 'string'
      || typeof candidate.updated_at !== 'string'
      || typeof candidate.message_count !== 'number'
    ) {
      continue;
    }
    const title =
      typeof candidate.title === 'string' && candidate.title.trim()
        ? candidate.title.trim()
        : candidate.session_id.slice(0, 8);
    sessions.push({
      session_id: candidate.session_id,
      title,
      created_at: candidate.created_at,
      updated_at: candidate.updated_at,
      message_count: candidate.message_count,
      chat_message_count:
        typeof candidate.chat_message_count === 'number'
          ? candidate.chat_message_count
          : candidate.message_count,
      workspace_message_count:
        typeof candidate.workspace_message_count === 'number'
          ? candidate.workspace_message_count
          : 0,
      last_message_lane:
        candidate.last_message_lane === 'workspace' || candidate.last_message_lane === 'chat'
          ? candidate.last_message_lane
          : null,
      title_override: typeof candidate.title_override === 'string' ? candidate.title_override : null,
      folder_id: typeof candidate.folder_id === 'string' ? candidate.folder_id : null,
    });
  }
  return sessions;
};

export const parseFolders = (value: unknown): FolderSummary[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const folders: FolderSummary[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as {
      folder_id?: unknown;
      name?: unknown;
      created_at?: unknown;
      updated_at?: unknown;
    };
    if (
      typeof candidate.folder_id !== 'string'
      || typeof candidate.name !== 'string'
      || typeof candidate.created_at !== 'string'
      || typeof candidate.updated_at !== 'string'
    ) {
      continue;
    }
    folders.push({
      folder_id: candidate.folder_id,
      name: candidate.name,
      created_at: candidate.created_at,
      updated_at: candidate.updated_at,
    });
  }
  return folders;
};

export const parseSessionOutput = (value: unknown): { content: string | null; updatedAt: string | null } => {
  if (!value || typeof value !== 'object') {
    return { content: null, updatedAt: null };
  }
  const candidate = value as { content?: unknown; updated_at?: unknown };
  const content = typeof candidate.content === 'string' ? candidate.content : null;
  const updatedAt = typeof candidate.updated_at === 'string' ? candidate.updated_at : null;
  return { content, updatedAt };
};

export const parseSessionFiles = (value: unknown): string[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
};

export const parseSessionArtifacts = (value: unknown): SessionArtifactRecord[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const artifacts: SessionArtifactRecord[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as {
      id?: unknown;
      kind?: unknown;
      title?: unknown;
      content?: unknown;
      created_at?: unknown;
      display_target?: unknown;
      artifact_kind?: unknown;
      file_name?: unknown;
      file_ext?: unknown;
      language?: unknown;
      file_content?: unknown;
    };
    if (typeof candidate.id !== 'string' || !candidate.id.trim()) {
      continue;
    }
    if (candidate.kind !== 'output') {
      continue;
    }
    if (typeof candidate.content !== 'string' || !candidate.content.trim()) {
      continue;
    }
    const title =
      typeof candidate.title === 'string' && candidate.title.trim()
        ? candidate.title.trim()
        : candidate.content.split('\n').find((line) => line.trim())?.trim().slice(0, 80) || 'Result';
    artifacts.push({
      id: candidate.id.trim(),
      kind: 'output',
      title,
      content: candidate.content,
      createdAt: typeof candidate.created_at === 'string' ? candidate.created_at : null,
      displayTarget: candidate.display_target === 'canvas' ? 'canvas' : 'chat',
      artifactKind: candidate.artifact_kind === 'file' ? 'file' : 'text',
      fileName: typeof candidate.file_name === 'string' ? candidate.file_name : null,
      fileExt: typeof candidate.file_ext === 'string' ? candidate.file_ext : null,
      language: typeof candidate.language === 'string' ? candidate.language : null,
      fileContent: typeof candidate.file_content === 'string' ? candidate.file_content : null,
    });
  }
  return artifacts;
};

export const parseDisplayDecision = (value: unknown): DisplayDecision | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const candidate = value as { target?: unknown; artifact_id?: unknown; forced?: unknown };
  if (candidate.target !== 'chat' && candidate.target !== 'canvas') {
    return null;
  }
  return {
    target: candidate.target,
    artifactId: typeof candidate.artifact_id === 'string' ? candidate.artifact_id : null,
    forced: candidate.forced === true,
  };
};

export const parseUiDecision = (value: unknown): UiDecision | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const candidate = value as {
    id?: unknown;
    kind?: unknown;
    decision_type?: unknown;
    status?: unknown;
    blocking?: unknown;
    reason?: unknown;
    summary?: unknown;
    proposed_action?: unknown;
    options?: unknown;
    default_option_id?: unknown;
    context?: unknown;
    created_at?: unknown;
    updated_at?: unknown;
    resolved_at?: unknown;
  };
  if (typeof candidate.id !== 'string' || !candidate.id.trim()) {
    return null;
  }
  if (candidate.kind !== 'approval' && candidate.kind !== 'decision') {
    return null;
  }
  if (
    candidate.status !== 'pending'
    && candidate.status !== 'approved'
    && candidate.status !== 'rejected'
    && candidate.status !== 'executing'
    && candidate.status !== 'resolved'
  ) {
    return null;
  }
  if (typeof candidate.reason !== 'string' || typeof candidate.summary !== 'string') {
    return null;
  }
  if (typeof candidate.created_at !== 'string' || typeof candidate.updated_at !== 'string') {
    return null;
  }
  const optionsRaw = candidate.options;
  const options = Array.isArray(optionsRaw)
    ? optionsRaw
        .filter((item): item is { id: string; title: string; action: string; payload?: unknown; risk?: unknown } => {
          return (
            !!item
            && typeof item === 'object'
            && typeof (item as { id?: unknown }).id === 'string'
            && typeof (item as { title?: unknown }).title === 'string'
            && typeof (item as { action?: unknown }).action === 'string'
          );
        })
        .map((item) => ({
          id: item.id,
          title: item.title,
          action: item.action,
          payload: item.payload && typeof item.payload === 'object' ? (item.payload as Record<string, unknown>) : {},
          risk: typeof item.risk === 'string' ? item.risk : 'low',
        }))
    : [];
  const proposedAction =
    candidate.proposed_action && typeof candidate.proposed_action === 'object'
      ? (candidate.proposed_action as Record<string, unknown>)
      : {};
  const context =
    candidate.context && typeof candidate.context === 'object'
      ? (candidate.context as Record<string, unknown>)
      : {};
  return {
    id: candidate.id.trim(),
    kind: candidate.kind,
    decision_type:
      candidate.decision_type === 'tool_approval'
      || candidate.decision_type === 'plan_execute'
      || candidate.decision_type === 'agent_decision'
        ? candidate.decision_type
        : null,
    status: candidate.status,
    blocking: candidate.blocking === true,
    reason: candidate.reason,
    summary: candidate.summary,
    proposed_action: proposedAction,
    options,
    default_option_id:
      typeof candidate.default_option_id === 'string' ? candidate.default_option_id : null,
    context,
    created_at: candidate.created_at,
    updated_at: candidate.updated_at,
    resolved_at: typeof candidate.resolved_at === 'string' ? candidate.resolved_at : null,
  };
};

export const parseSessionMode = (value: unknown): SessionMode => {
  if (isSessionMode(value)) {
    return value;
  }
  return 'ask';
};

export const parseAutoState = (value: unknown): AutoState | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const candidate = value as {
    run_id?: unknown;
    status?: unknown;
    goal?: unknown;
    pool_size?: unknown;
    started_at?: unknown;
    updated_at?: unknown;
    planner?: unknown;
    plan?: unknown;
    coders?: unknown;
    merge?: unknown;
    verifier?: unknown;
    approval?: unknown;
    error?: unknown;
  };
  if (
    typeof candidate.run_id !== 'string'
    || typeof candidate.status !== 'string'
    || typeof candidate.goal !== 'string'
    || typeof candidate.started_at !== 'string'
    || typeof candidate.updated_at !== 'string'
  ) {
    return null;
  }
  const status = candidate.status;
  const isKnownStatus =
    status === 'idle'
    || status === 'planning'
    || status === 'coding'
    || status === 'merging'
    || status === 'verifying'
    || status === 'waiting_approval'
    || status === 'completed'
    || status === 'failed_conflict'
    || status === 'failed_verifier'
    || status === 'failed_worker'
    || status === 'failed_internal'
    || status === 'cancelled';
  if (!isKnownStatus) {
    return null;
  }
  const poolSize = typeof candidate.pool_size === 'number' && Number.isFinite(candidate.pool_size)
    ? Math.max(1, Math.floor(candidate.pool_size))
    : 1;
  return {
    run_id: candidate.run_id,
    status,
    goal: candidate.goal,
    pool_size: poolSize,
    started_at: candidate.started_at,
    updated_at: candidate.updated_at,
    planner: candidate.planner && typeof candidate.planner === 'object'
      ? (candidate.planner as Record<string, unknown>)
      : {},
    plan: candidate.plan && typeof candidate.plan === 'object'
      ? (candidate.plan as Record<string, unknown>)
      : null,
    coders: Array.isArray(candidate.coders)
      ? candidate.coders.filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
      : [],
    merge: candidate.merge && typeof candidate.merge === 'object'
      ? (candidate.merge as Record<string, unknown>)
      : {},
    verifier: candidate.verifier && typeof candidate.verifier === 'object'
      ? (candidate.verifier as Record<string, unknown>)
      : null,
    approval: candidate.approval && typeof candidate.approval === 'object'
      ? (candidate.approval as Record<string, unknown>)
      : null,
    error: typeof candidate.error === 'string' ? candidate.error : null,
  };
};

export const parsePlanEnvelope = (value: unknown): PlanEnvelope | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const candidate = value as {
    plan_id?: unknown;
    plan_hash?: unknown;
    plan_revision?: unknown;
    status?: unknown;
    goal?: unknown;
    scope_in?: unknown;
    scope_out?: unknown;
    assumptions?: unknown;
    inputs_needed?: unknown;
    audit_log?: unknown;
    steps?: unknown;
    exit_criteria?: unknown;
    created_at?: unknown;
    updated_at?: unknown;
    approved_at?: unknown;
    approved_by?: unknown;
  };
  if (
    typeof candidate.plan_id !== 'string'
    || typeof candidate.plan_hash !== 'string'
    || typeof candidate.goal !== 'string'
    || typeof candidate.created_at !== 'string'
    || typeof candidate.updated_at !== 'string'
  ) {
    return null;
  }
  const status =
    candidate.status === 'draft'
    || candidate.status === 'approved'
    || candidate.status === 'running'
    || candidate.status === 'completed'
    || candidate.status === 'failed'
    || candidate.status === 'cancelled'
      ? candidate.status
      : 'draft';
  const planRevision =
    typeof candidate.plan_revision === 'number'
    && Number.isFinite(candidate.plan_revision)
    && candidate.plan_revision > 0
      ? Math.floor(candidate.plan_revision)
      : 1;
  const normalizeStringList = (input: unknown): string[] =>
    Array.isArray(input)
      ? input.filter((item): item is string => typeof item === 'string')
      : [];
  const stepsRaw = Array.isArray(candidate.steps) ? candidate.steps : [];
  const steps = stepsRaw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
    .map((step) => {
      const rawStatus = step.status;
      const stepStatus: PlanStepStatus =
        rawStatus === 'todo'
        || rawStatus === 'doing'
        || rawStatus === 'waiting_approval'
        || rawStatus === 'blocked'
        || rawStatus === 'done'
        || rawStatus === 'failed'
          ? rawStatus
          : 'todo';
      return {
        step_id: typeof step.step_id === 'string' ? step.step_id : '',
        title: typeof step.title === 'string' ? step.title : '',
        description: typeof step.description === 'string' ? step.description : '',
        allowed_tool_kinds: normalizeStringList(step.allowed_tool_kinds),
        acceptance_checks: normalizeStringList(step.acceptance_checks),
        status: stepStatus,
        details: typeof step.details === 'string' ? step.details : null,
      };
    })
    .filter((step) => step.step_id.trim().length > 0);
  return {
    plan_id: candidate.plan_id,
    plan_hash: candidate.plan_hash,
    plan_revision: planRevision,
    status,
    goal: candidate.goal,
    scope_in: normalizeStringList(candidate.scope_in),
    scope_out: normalizeStringList(candidate.scope_out),
    assumptions: normalizeStringList(candidate.assumptions),
    inputs_needed: normalizeStringList(candidate.inputs_needed),
    audit_log: Array.isArray(candidate.audit_log) ? candidate.audit_log : [],
    steps,
    exit_criteria: normalizeStringList(candidate.exit_criteria),
    created_at: candidate.created_at,
    updated_at: candidate.updated_at,
    approved_at: typeof candidate.approved_at === 'string' ? candidate.approved_at : null,
    approved_by: typeof candidate.approved_by === 'string' ? candidate.approved_by : null,
  };
};

export const parseTaskExecution = (value: unknown): TaskExecutionState | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const candidate = value as {
    task_id?: unknown;
    plan_id?: unknown;
    plan_hash?: unknown;
    current_step_id?: unknown;
    status?: unknown;
    started_at?: unknown;
    updated_at?: unknown;
  };
  if (
    typeof candidate.task_id !== 'string'
    || typeof candidate.plan_id !== 'string'
    || typeof candidate.plan_hash !== 'string'
    || typeof candidate.started_at !== 'string'
    || typeof candidate.updated_at !== 'string'
  ) {
    return null;
  }
  const status =
    candidate.status === 'running'
    || candidate.status === 'completed'
    || candidate.status === 'failed'
    || candidate.status === 'cancelled'
      ? candidate.status
      : 'running';
  return {
    task_id: candidate.task_id,
    plan_id: candidate.plan_id,
    plan_hash: candidate.plan_hash,
    current_step_id: typeof candidate.current_step_id === 'string' ? candidate.current_step_id : null,
    status,
    started_at: candidate.started_at,
    updated_at: candidate.updated_at,
  };
};

export const parseTraceId = (value: unknown): string | null => {
  if (typeof value !== 'string') {
    return null;
  }
  const normalized = value.trim();
  return normalized || null;
};

export const parseMwvReport = (value: unknown): MwvReportUi | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const report = value as Record<string, unknown>;
  const normalized: MwvReportUi = {};

  if (typeof report.route === 'string') {
    normalized.route = report.route;
  }
  if (report.trace_id === null || typeof report.trace_id === 'string') {
    normalized.trace_id = report.trace_id;
  }
  if (typeof report.stop_reason_code === 'string') {
    normalized.stop_reason_code = report.stop_reason_code;
  }
  if (typeof report.plan_summary === 'string') {
    normalized.plan_summary = report.plan_summary;
  }
  if (typeof report.execution_summary === 'string') {
    normalized.execution_summary = report.execution_summary;
  }
  if (report.attempts && typeof report.attempts === 'object' && !Array.isArray(report.attempts)) {
    const attempts = report.attempts as { current?: unknown; max?: unknown };
    normalized.attempts = {
      current: typeof attempts.current === 'number' ? attempts.current : undefined,
      max: typeof attempts.max === 'number' ? attempts.max : undefined,
    };
  }
  if (report.verifier && typeof report.verifier === 'object' && !Array.isArray(report.verifier)) {
    normalized.verifier = report.verifier as { status?: string; duration_ms?: number | null; [k: string]: unknown };
  }

  for (const [key, entry] of Object.entries(report)) {
    if (!(key in normalized)) {
      normalized[key] = entry;
    }
  }
  return normalized;
};

export const toMessageRuntimeMeta = (
  message: ChatMessage,
  lane: MessageLane,
  previous: MessageRuntimeMeta | null,
): MessageRuntimeMeta => {
  return {
    messageId: message.message_id,
    lane,
    traceId: previous?.traceId ?? message.trace_id ?? null,
    isFinal: true,
    mwvReport: previous?.mwvReport ?? null,
  };
};

export const groupSessionByDate = (value: string): 'today' | 'yesterday' | 'older' => {
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) {
    return 'older';
  }
  const date = new Date(parsed);
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfYesterday = new Date(startOfToday);
  startOfYesterday.setDate(startOfYesterday.getDate() - 1);
  if (date >= startOfToday) {
    return 'today';
  }
  if (date >= startOfYesterday) {
    return 'yesterday';
  }
  return 'older';
};

export const buildCanvasMessages = (
  messages: ChatMessage[],
  lane: MessageLane,
  runtimeMetaByMessageId: Record<string, MessageRuntimeMeta>,
): CanvasMessage[] => {
  return messages
    .filter(
      (message): message is ChatMessage & { role: 'user' | 'assistant' } =>
        message.role === 'user' || message.role === 'assistant',
    )
    .map((message) => ({
      id: message.message_id,
      messageId: message.message_id,
      role: message.role,
      content: message.content,
      createdAt: message.created_at,
      traceId: message.trace_id,
      parentUserMessageId: message.parent_user_message_id,
      attachments: message.attachments,
      transient: false,
      runtimeMeta: runtimeMetaByMessageId[message.message_id]
        ?? toMessageRuntimeMeta(message, lane, null),
    }));
};

const inferArtifactType = (content: string): Artifact['type'] => {
  if (content.includes('```')) {
    const match = content.match(/```\s*([a-zA-Z0-9_-]+)/);
    const lang = match ? match[1].toLowerCase() : '';
    if (lang === 'python' || lang === 'py') return 'PY';
    if (lang === 'javascript' || lang === 'js') return 'JS';
    if (lang === 'typescript' || lang === 'ts') return 'TS';
    if (lang === 'json') return 'JSON';
    if (lang === 'html') return 'HTML';
    if (lang === 'css') return 'CSS';
    if (lang === 'sh' || lang === 'shell' || lang === 'bash') return 'SH';
    return 'MD';
  }
  return 'TXT';
};

const inferArtifactTypeFromPath = (path: string): Artifact['type'] => {
  const normalized = path.trim().toLowerCase();
  if (normalized.endsWith('.py')) return 'PY';
  if (normalized.endsWith('.ts') || normalized.endsWith('.tsx')) return 'TS';
  if (normalized.endsWith('.js') || normalized.endsWith('.jsx')) return 'JS';
  if (normalized.endsWith('.json')) return 'JSON';
  if (normalized.endsWith('.html')) return 'HTML';
  if (normalized.endsWith('.css')) return 'CSS';
  if (normalized.endsWith('.sh')) return 'SH';
  if (normalized.endsWith('.md') || normalized.endsWith('.markdown')) return 'MD';
  return 'TXT';
};

const inferArtifactTypeFromLanguage = (languageRaw: string): Artifact['type'] => {
  const language = languageRaw.trim().toLowerCase();
  if (!language) {
    return 'TXT';
  }
  if (language === 'python' || language === 'py') return 'PY';
  if (language === 'typescript' || language === 'ts' || language === 'tsx') return 'TS';
  if (language === 'javascript' || language === 'js' || language === 'jsx') return 'JS';
  if (language === 'json') return 'JSON';
  if (language === 'html') return 'HTML';
  if (language === 'css') return 'CSS';
  if (language === 'shell' || language === 'bash' || language === 'sh') return 'SH';
  if (language === 'md' || language === 'markdown') return 'MD';
  return 'TXT';
};

const inferArtifactCategory = (type: Artifact['type']): Artifact['category'] => {
  if (type === 'TXT' || type === 'MD') return 'Document';
  if (type === 'JSON') return 'Config';
  if (type === 'SH') return 'Script';
  return 'Code';
};

export const buildArtifactsFromSources = (
  artifactsHistory: SessionArtifactRecord[],
  files: string[],
  streamingContentByArtifactId: Record<string, string>,
): Artifact[] => {
  const artifacts: Artifact[] = [];
  const seen = new Set<string>();
  const seenFileNames = new Set<string>();

  for (const item of artifactsHistory) {
    if (!item.content.trim()) {
      continue;
    }
    const id = `output-${item.id}`;
    const hasStreamOverride = Object.prototype.hasOwnProperty.call(streamingContentByArtifactId, id);
    const content = hasStreamOverride ? streamingContentByArtifactId[id] : item.content;
    let type: Artifact['type'];
    if (item.fileExt) {
      type = inferArtifactTypeFromPath(`file.${item.fileExt}`);
    } else if (item.fileName) {
      type = inferArtifactTypeFromPath(item.fileName);
    } else if (item.language) {
      type = inferArtifactTypeFromLanguage(item.language);
    } else {
      type = inferArtifactType(item.content);
    }
    if (seen.has(id)) {
      continue;
    }
    seen.add(id);
    const normalizedFileName = item.fileName?.trim().toLowerCase() ?? '';
    if (normalizedFileName) {
      seenFileNames.add(normalizedFileName);
    }
    artifacts.push({
      id,
      name: item.fileName || item.title,
      type,
      category: inferArtifactCategory(type),
      content,
      artifactKind: item.artifactKind,
      sourceArtifactId: item.id,
      fileName: item.fileName,
      fileExt: item.fileExt,
      language: item.language,
      fileContent: item.fileContent,
    });
  }

  for (const rawPath of files) {
    const path = rawPath.trim();
    const fileId = `file-${path}`;
    if (!path || seen.has(fileId)) {
      continue;
    }
    const pathBaseName = path.split('/').pop()?.toLowerCase() ?? '';
    if (pathBaseName && seenFileNames.has(pathBaseName)) {
      continue;
    }
    seen.add(fileId);
    const type = inferArtifactTypeFromPath(path);
    artifacts.push({
      id: fileId,
      name: path,
      type,
      category: inferArtifactCategory(type),
      content: `File path: ${path}`,
      artifactKind: 'file',
      sessionFilePath: path,
      fileName: path.split('/').pop() ?? path,
    });
  }

  return artifacts;
};

export const extractFilenameFromDisposition = (
  disposition: string | null,
  fallbackName: string,
): string => {
  if (!disposition) {
    return fallbackName;
  }
  const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match && utf8Match[1]) {
    try {
      const decoded = decodeURIComponent(utf8Match[1].trim());
      return decoded || fallbackName;
    } catch {
      return utf8Match[1].trim() || fallbackName;
    }
  }
  const basicMatch = disposition.match(/filename="?([^";]+)"?/i);
  if (basicMatch && basicMatch[1]) {
    return basicMatch[1].trim() || fallbackName;
  }
  return fallbackName;
};

export const triggerBrowserDownload = (blob: Blob, filename: string): void => {
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(url);
};

export const extractSessionIdFromPayload = (payload: unknown): string | null => {
  if (!payload || typeof payload !== 'object') {
    return null;
  }
  const maybeSession = payload as {
    session_id?: unknown;
    session?: { session_id?: unknown };
  };
  if (typeof maybeSession.session_id === 'string' && maybeSession.session_id.trim()) {
    return maybeSession.session_id.trim();
  }
  const nested = maybeSession.session;
  if (nested && typeof nested === 'object' && typeof nested.session_id === 'string') {
    return nested.session_id.trim() || null;
  }
  return null;
};
