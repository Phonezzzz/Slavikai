import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ClipboardEvent,
  type KeyboardEvent,
} from "react";
import {
  Send,
  Copy,
  Edit2,
  RefreshCcw,
  Volume2,
  Pause,
  ThumbsUp,
  ThumbsDown,
  Paperclip,
  Mic,
  Globe2,
  Plus,
  Check,
  PanelRight,
  SlidersHorizontal,
  LoaderCircle,
  X,
  FileText,
  Square,
} from "lucide-react";

import BrainLogo from "../../assets/brain.png";
import { MessageRenderer } from "../../features/messages";
import type { RenderableMessage } from "../../features/messages";
import { TtsAudioPlayer, useTtsAudioPlayer } from "../../features/audio";
import type { DecisionRespondChoice, MessageRuntimeMeta, SessionMode, UiDecision } from "../types";
import type { ToolActivity } from "../../features/messages/types";
import { getDecisionDisplayState } from "../decision-display";
import { DecisionPanel } from "./decision-panel";
import {
  MAX_COMPOSER_ATTACHMENTS,
  buildPastedTextAttachment,
  createComposerAttachmentId,
  readComposerAttachmentFromFile,
} from "../../features/composer/attachment-utils";

// ====== Types ======

export interface CanvasMessage {
  id: string;
  messageId: string;
  role: "user" | "assistant";
  content: string;
  createdAt?: string;
  traceId?: string | null;
  parentUserMessageId?: string | null;
  attachments?: Array<{ name: string; mime: string; content: string }>;
  transient?: boolean;
  runtimeMeta?: MessageRuntimeMeta | null;
  toolActivity?: ToolActivity[] | null;
}

export type CanvasComposerAttachment = {
  name: string;
  mime: string;
  content: string;
};

export type CanvasSendPayload = {
  content: string;
  attachments?: CanvasComposerAttachment[];
  webSearch?: boolean;
  regenerateLast?: boolean;
};

export async function deliverTranscription(
  text: string,
  runtimeMode: SessionMode,
  onSendMessage: CanvasProps["onSendMessage"],
  insertIntoComposer: (value: string) => void,
): Promise<"sent" | "composed"> {
  const normalized = text.trim();
  if (!normalized) {
    throw new Error("STT returned empty text.");
  }
  if (runtimeMode === "desktop" && onSendMessage) {
    const sent = await onSendMessage({ content: normalized, attachments: [] });
    if (sent === false) {
      throw new Error("Desktop voice request was not accepted.");
    }
    return "sent";
  }
  insertIntoComposer(text);
  return "composed";
}

interface CanvasProps {
  messages?: CanvasMessage[];
  pendingMessage?: CanvasMessage | null;
  streamingAssistantMessage?: CanvasMessage | null;
  sending?: boolean;
  cancelling?: boolean;
  onSendMessage?: (payload: CanvasSendPayload) => Promise<boolean> | boolean | void;
  onCancelSend?: () => Promise<boolean> | boolean | void;
  onSendFeedback?: (interactionId: string, rating: "good" | "bad") => Promise<boolean>;
  className?: string;
  modelName?: string;
  modelProvider?: string | null;
  onOpenSessionDrawer?: () => void;
  statusMessage?: string | null;
  forceCanvasNext?: boolean;
  onToggleForceCanvasNext?: () => void;
  longPasteToFileEnabled?: boolean;
  longPasteThresholdChars?: number;
  decision?: UiDecision | null;
  decisionBusy?: boolean;
  decisionError?: string | null;
  runtimeMode?: SessionMode;
  onDecisionRespond?: (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
  ) => Promise<void> | void;
}

// ====== Sub Components ======

function MessageBubble({ message }: { message: CanvasMessage }) {
  const renderable: RenderableMessage = {
    kind: "message",
    message,
    meta: message.runtimeMeta ?? null,
    toolActivity: message.toolActivity ?? null,
  };

  return (
    <div className={`flex gap-3 ${message.role === "user" ? "flex-row-reverse" : ""}`}>
      <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-[#2a2a30] border border-[#3a3a42]">
        {message.role === "assistant" ? (
          <img
            src={BrainLogo}
            alt="SlavikAI"
            className="w-4 h-4 object-contain"
          />
        ) : (
          <span className="text-[11px] font-semibold text-[#aeb1ff]">YOU</span>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <MessageRenderer context="chat" message={renderable} />
      </div>
    </div>
  );
}

function MessageActions({
  message,
  copied,
  speaking,
  feedbackRating,
  feedbackBusy,
  sending,
  canRefresh,
  canFeedback,
  onCopy,
  onEdit,
  onRefresh,
  onListenToggle,
  onLike,
  onDislike,
}: {
  message: CanvasMessage;
  copied: boolean;
  speaking: boolean;
  feedbackRating: "good" | "bad" | null;
  feedbackBusy: boolean;
  sending: boolean;
  canRefresh: boolean;
  canFeedback: boolean;
  onCopy: () => void;
  onEdit: () => void;
  onRefresh: () => void;
  onListenToggle: () => void;
  onLike: () => void;
  onDislike: () => void;
}) {
  const isUser = message.role === "user";
  const baseClass =
    "rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40";

  return (
    <div className={`mt-2 flex items-center gap-1 ${isUser ? "justify-end" : "justify-start"}`}>
      <button
        type="button"
        onClick={onCopy}
        className={baseClass}
        title="Copy"
        aria-label="Copy"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
      </button>

      {isUser ? (
        <button
          type="button"
          onClick={onEdit}
          className={baseClass}
          title="Edit"
          aria-label="Edit"
        >
          <Edit2 className="h-3.5 w-3.5" />
        </button>
      ) : null}

      {!isUser ? (
        <>
          <button
            type="button"
            onClick={onRefresh}
            disabled={!canRefresh || sending}
            className={baseClass}
            title={canRefresh ? "Refresh" : "Refresh unavailable"}
            aria-label="Refresh"
          >
            <RefreshCcw className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={onListenToggle}
            className={baseClass}
            title={speaking ? "Pause listen" : "Listen"}
            aria-label={speaking ? "Pause listen" : "Listen"}
          >
            {speaking ? <Pause className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
          </button>
          <button
            type="button"
            onClick={onLike}
            disabled={!canFeedback || feedbackBusy}
            className={`${baseClass} ${feedbackRating === "good" ? "text-emerald-300" : ""}`}
            title={canFeedback ? "Like" : "Like unavailable"}
            aria-label="Like"
          >
            <ThumbsUp className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={onDislike}
            disabled={!canFeedback || feedbackBusy}
            className={`${baseClass} ${feedbackRating === "bad" ? "text-rose-300" : ""}`}
            title={canFeedback ? "Dislike" : "Dislike unavailable"}
            aria-label="Dislike"
          >
            <ThumbsDown className="h-3.5 w-3.5" />
          </button>
        </>
      ) : null}
    </div>
  );
}

// ====== Main Canvas Component ======

export function Canvas({
  messages = [],
  pendingMessage = null,
  streamingAssistantMessage = null,
  sending = false,
  cancelling = false,
  onSendMessage,
  onCancelSend,
  onSendFeedback,
  className = "",
  modelName = "Model not selected",
  modelProvider = null,
  onOpenSessionDrawer,
  statusMessage = null,
  forceCanvasNext = false,
  onToggleForceCanvasNext,
  longPasteToFileEnabled = true,
  longPasteThresholdChars = 12000,
  decision = null,
  decisionBusy = false,
  decisionError = null,
  onDecisionRespond,
  runtimeMode = "ask",
}: CanvasProps) {
  const [inputValue, setInputValue] = useState("");
  const [composerAttachments, setComposerAttachments] = useState<
    Array<CanvasComposerAttachment & { id: string }>
  >([]);
  const [pasteUndo, setPasteUndo] = useState<{
    attachmentId: string;
    originalText: string;
  } | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [feedbackByMessageId, setFeedbackByMessageId] = useState<Record<string, "good" | "bad">>(
    {},
  );
  const [feedbackBusyMessageId, setFeedbackBusyMessageId] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [sttError, setSttError] = useState<string | null>(null);
  const [webSearchNext, setWebSearchNext] = useState(false);
  const [actionsMenuOpen, setActionsMenuOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const caretSelectionRef = useRef<{ start: number; end: number }>({ start: 0, end: 0 });
  const ttsPlayer = useTtsAudioPlayer();
  const displayMessages = useMemo(() => {
    const items = [...messages];
    if (pendingMessage) {
      items.push(pendingMessage);
    }
    if (streamingAssistantMessage) {
      items.push(streamingAssistantMessage);
    }
    return items;
  }, [messages, pendingMessage, streamingAssistantMessage]);
  const latestAssistantMessageId = useMemo(
    () => [...messages].reverse().find((item) => item.role === "assistant")?.messageId ?? null,
    [messages],
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [displayMessages]);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    const lineHeightPx = 24;
    const maxHeight = lineHeightPx * 5;
    textarea.style.height = "auto";
    const nextHeight = Math.min(Math.max(textarea.scrollHeight, lineHeightPx), maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [inputValue]);

  const canUseMediaRecorder = useMemo(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return (
      typeof window.MediaRecorder !== "undefined"
      && !!window.navigator?.mediaDevices
      && typeof window.navigator.mediaDevices.getUserMedia === "function"
    );
  }, []);

  const effectiveLongPasteThreshold = useMemo(() => {
    const normalized = Number.isFinite(longPasteThresholdChars)
      ? Math.floor(longPasteThresholdChars)
      : 12000;
    return Math.max(1000, Math.min(80000, normalized));
  }, [longPasteThresholdChars]);
  const decisionState = getDecisionDisplayState(decision, decisionBusy, decisionError);
  const composerBlocked = decisionState.isBlocking;
  const webSearchAvailable = modelProvider !== null && modelProvider.trim().length > 0;
  const actionsDisabled = sending || isTranscribing || composerBlocked;
  const isCanvasStatus =
    typeof statusMessage === "string"
    && /canvas/i.test(statusMessage)
    && /(result|результат|сформирован|открыт)/i.test(statusMessage);
  const visibleStatusMessage = isCanvasStatus ? null : statusMessage;

  const toComposerAttachments = (
    items: Array<{ name: string; mime: string; content: string }> | undefined,
  ): Array<CanvasComposerAttachment & { id: string }> => {
    if (!items || items.length === 0) {
      return [];
    }
    return items.map((item, index) => ({
      id: `composer-${Date.now()}-${index}`,
      name: item.name,
      mime: item.mime,
      content: item.content,
    }));
  };

  const buildMessageTextForCopy = (message: CanvasMessage): string => {
    const attachments = message.attachments ?? [];
    if (attachments.length === 0) {
      return message.content;
    }
    const lines: string[] = [];
    if (message.content.trim()) {
      lines.push(message.content.trim());
      lines.push("");
    }
    lines.push("[attachments]");
    attachments.forEach((attachment, index) => {
      lines.push(`#${index + 1} ${attachment.name} (${attachment.mime})`);
      lines.push(attachment.content);
      lines.push("---");
    });
    return lines.join("\n");
  };

  const insertTextIntoComposer = (rawText: string) => {
    const text = rawText.trim();
    if (!text) {
      return;
    }
    const textarea = textareaRef.current;
    const isActive =
      typeof document !== "undefined"
      && textarea !== null
      && document.activeElement === textarea;
    let nextCaret = 0;
    setInputValue((prev) => {
      const startBase = isActive
        ? textarea?.selectionStart ?? prev.length
        : prev.length;
      const endBase = isActive
        ? textarea?.selectionEnd ?? prev.length
        : prev.length;
      const start = Math.max(0, Math.min(startBase, prev.length));
      const end = Math.max(start, Math.min(endBase, prev.length));
      const prefix = prev.slice(0, start);
      const suffix = prev.slice(end);
      const spacerBefore = prefix.length > 0 && !/\s$/.test(prefix) ? " " : "";
      const spacerAfter = suffix.length > 0 && !/^\s/.test(suffix) ? " " : "";
      const insertion = `${spacerBefore}${text}${spacerAfter}`;
      nextCaret = prefix.length + insertion.length;
      return `${prefix}${insertion}${suffix}`;
    });
    if (isActive && textarea) {
      window.requestAnimationFrame(() => {
        textarea.focus();
        textarea.setSelectionRange(nextCaret, nextCaret);
        caretSelectionRef.current = { start: nextCaret, end: nextCaret };
      });
    }
  };

  const extractErrorMessage = (payload: unknown, fallback: string): string => {
    if (!payload || typeof payload !== "object") {
      return fallback;
    }
    const body = payload as { error?: { message?: unknown } };
    if (body.error && typeof body.error.message === "string" && body.error.message.trim()) {
      return body.error.message;
    }
    return fallback;
  };

  const pushComposerAttachments = (attachments: CanvasComposerAttachment[]): boolean => {
    if (attachments.length === 0) {
      return true;
    }
    let appended = false;
    let truncated = false;
    setComposerAttachments((prev) => {
      const remaining = MAX_COMPOSER_ATTACHMENTS - prev.length;
      if (remaining <= 0) {
        truncated = true;
        return prev;
      }
      const nextItems = attachments.slice(0, remaining).map((attachment) => ({
        id: createComposerAttachmentId("canvas-attachment"),
        ...attachment,
      }));
      truncated = attachments.length > remaining;
      appended = nextItems.length > 0;
      return [...prev, ...nextItems];
    });
    if (truncated) {
      setSttError("Достигнут лимит вложений в одном сообщении.");
    }
    return appended;
  };

  const appendFilesToComposer = async (files: File[]) => {
    if (files.length === 0) {
      return;
    }
    setSttError(null);
    try {
      const attachments = await Promise.all(files.map((file) => readComposerAttachmentFromFile(file)));
      pushComposerAttachments(attachments);
    } catch (error) {
      setSttError(error instanceof Error ? error.message : "Не удалось подготовить вложение.");
    }
  };

  const handleAttachFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    await appendFilesToComposer(files);
  };

  const handleOpenFilePicker = () => {
    setActionsMenuOpen(false);
    fileInputRef.current?.click();
  };

  const handleToggleWebSearch = () => {
    if (!webSearchAvailable) {
      return;
    }
    setWebSearchNext((prev) => !prev);
    setActionsMenuOpen(false);
  };

  const handleToggleCanvasNext = () => {
    onToggleForceCanvasNext?.();
    setActionsMenuOpen(false);
  };

  const handleSend = async () => {
    if (sending || isTranscribing) {
      return;
    }
    const trimmed = inputValue.trim();
    const attachmentsPayload: CanvasComposerAttachment[] = composerAttachments.map((item) => ({
      name: item.name,
      mime: item.mime,
      content: item.content,
    }));
    if (!trimmed && attachmentsPayload.length === 0) {
      return;
    }
    const previousInputValue = inputValue;
    const previousAttachments = composerAttachments;
    const previousPasteUndo = pasteUndo;
    const previousWebSearchNext = webSearchNext;
    setInputValue("");
    setComposerAttachments([]);
    setPasteUndo(null);
    setSttError(null);
    setWebSearchNext(false);
    setActionsMenuOpen(false);

    const sent = await onSendMessage?.({
      content: trimmed,
      attachments: attachmentsPayload.length > 0 ? attachmentsPayload : undefined,
      webSearch: previousWebSearchNext && webSearchAvailable ? true : undefined,
    });
    if (sent === false) {
      setInputValue((current) => (current.trim().length === 0 ? previousInputValue : current));
      setComposerAttachments((current) => (current.length === 0 ? previousAttachments : current));
      setPasteUndo((current) => current ?? previousPasteUndo);
      setWebSearchNext(previousWebSearchNext);
      return;
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  const handleCopyMessage = async (message: CanvasMessage) => {
    try {
      await navigator.clipboard.writeText(buildMessageTextForCopy(message));
      setCopiedMessageId(message.messageId);
      window.setTimeout(() => {
        setCopiedMessageId((prev) => (prev === message.messageId ? null : prev));
      }, 1200);
    } catch {
      setCopiedMessageId(null);
    }
  };

  const handleEditMessage = (message: CanvasMessage) => {
    setInputValue(message.content);
    setComposerAttachments(toComposerAttachments(message.attachments));
    textareaRef.current?.focus();
  };

  const handleRefreshMessage = (message: CanvasMessage) => {
    if (!message.parentUserMessageId) {
      return;
    }
    const source = messages.find(
      (entry) =>
        entry.role === "user"
        && entry.messageId === message.parentUserMessageId
        && !entry.transient
        && (entry.content.trim().length > 0 || (entry.attachments?.length ?? 0) > 0),
    );
    if (!source) {
      return;
    }
    void onSendMessage?.({
      content: source.content.trim(),
      attachments: source.attachments ?? [],
      regenerateLast: true,
    });
  };

  const handleFeedback = async (message: CanvasMessage, rating: "good" | "bad") => {
    const interactionId = typeof message.traceId === "string" ? message.traceId.trim() : "";
    if (!interactionId || !onSendFeedback) {
      return;
    }
    const previous = feedbackByMessageId[message.messageId] ?? null;
    if (previous === rating) {
      return;
    }
    setFeedbackBusyMessageId(message.messageId);
    setFeedbackByMessageId((prev) => ({ ...prev, [message.messageId]: rating }));
    const ok = await onSendFeedback(interactionId, rating);
    if (!ok) {
      setFeedbackByMessageId((prev) => {
        const next = { ...prev };
        if (previous === null) {
          delete next[message.messageId];
        } else {
          next[message.messageId] = previous;
        }
        return next;
      });
    }
    setFeedbackBusyMessageId((prev) => (prev === message.messageId ? null : prev));
  };

  const transcribeAudio = async (blob: Blob) => {
    setIsTranscribing(true);
    setSttError(null);
    try {
      const extension = blob.type.includes("ogg") ? "ogg" : "webm";
      const file = new File([blob], `recording.${extension}`, { type: blob.type || "audio/webm" });
      const body = new FormData();
      body.append("audio", file);
      body.append("language", "ru");
      const response = await fetch("/ui/api/stt/transcribe", {
        method: "POST",
        body,
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, "STT request failed."));
      }
      const text = (payload as { text?: unknown }).text;
      if (typeof text !== "string" || !text.trim()) {
        throw new Error("STT returned empty text.");
      }
      await deliverTranscription(text, runtimeMode, onSendMessage, insertTextIntoComposer);
    } catch (error) {
      setSttError(error instanceof Error ? error.message : "STT failed.");
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleToggleRecording = async () => {
    if (!canUseMediaRecorder || sending || isTranscribing) {
      return;
    }
    if (isRecording) {
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.stop();
      }
      setIsRecording(false);
      return;
    }

    try {
      const stream = await window.navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeCandidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/ogg",
      ];
      const selectedMime = mimeCandidates.find((candidate) => {
        if (typeof window.MediaRecorder.isTypeSupported !== "function") {
          return false;
        }
        return window.MediaRecorder.isTypeSupported(candidate);
      });
      const recorder = selectedMime
        ? new window.MediaRecorder(stream, { mimeType: selectedMime })
        : new window.MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onerror = () => {
        setSttError("Не удалось записать аудио.");
        setIsRecording(false);
      };
      recorder.onstop = () => {
        setIsRecording(false);
        const chunks = audioChunksRef.current;
        audioChunksRef.current = [];
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
        }
        mediaRecorderRef.current = null;
        if (chunks.length === 0) {
          return;
        }
        const audioBlob = new Blob(chunks, {
          type: recorder.mimeType || "audio/webm",
        });
        void transcribeAudio(audioBlob);
      };
      recorder.start();
      setSttError(null);
      setIsRecording(true);
    } catch {
      setSttError("Микрофон недоступен.");
      setIsRecording(false);
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
    }
  };

  const handlePaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const imageFiles = Array.from(event.clipboardData.items ?? [])
      .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter((file): file is File => file instanceof File);
    if (imageFiles.length > 0) {
      event.preventDefault();
      void appendFilesToComposer(imageFiles);
      return;
    }
    const text = event.clipboardData.getData("text/plain");
    if (!longPasteToFileEnabled || !text || text.length <= effectiveLongPasteThreshold) {
      return;
    }
    if (composerAttachments.length >= MAX_COMPOSER_ATTACHMENTS) {
      setSttError("Достигнут лимит вложений в одном сообщении.");
      return;
    }
    event.preventDefault();
    const attachmentId = createComposerAttachmentId("paste");
    const attachment = buildPastedTextAttachment(text);
    setComposerAttachments((prev) => [...prev, { id: attachmentId, ...attachment }]);
    setPasteUndo({ attachmentId, originalText: text });
  };

  const handleUndoPaste = () => {
    if (!pasteUndo) {
      return;
    }
    const targetId = pasteUndo.attachmentId;
    const text = pasteUndo.originalText;
    setComposerAttachments((prev) => prev.filter((item) => item.id !== targetId));
    setPasteUndo(null);
    insertTextIntoComposer(text);
  };

  const handleRemoveComposerAttachment = (attachmentId: string) => {
    setComposerAttachments((prev) => prev.filter((item) => item.id !== attachmentId));
    setPasteUndo((prev) => (prev?.attachmentId === attachmentId ? null : prev));
  };

  return (
    <div
      className={`flex h-full min-h-0 flex-col bg-transparent ${className}`}
    >
      {/* Model selector header */}
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <button
            onClick={onOpenSessionDrawer}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[#1f1f24] bg-[#141418] hover:border-[#2a2a30] transition-colors cursor-pointer"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 text-[#888]" />
            <span className="text-[13px] text-[#d0d0d6]">Session</span>
          </button>
          <span className="text-[13px] text-[#8f8f95]">{modelName}</span>
        </div>
      </div>

      {/* Messages area */}
      <div className="min-h-0 flex-1 overflow-y-auto" data-scrollbar="auto">
        <div className="max-w-5xl mx-auto px-6 py-6 space-y-8">
          {displayMessages.map((msg) => {
            const isSavedMessage = !msg.transient;
            const canFeedback =
              !!onSendFeedback
              && msg.role === "assistant"
              && typeof msg.traceId === "string"
              && msg.traceId.trim().length > 0;
            const feedbackRating = feedbackByMessageId[msg.messageId] ?? null;
            return (
              <div key={msg.id}>
                <MessageBubble message={msg} />
                {isSavedMessage ? (
                  <MessageActions
                    message={msg}
                    copied={copiedMessageId === msg.messageId}
                    speaking={
                      ttsPlayer.state.activeMessageId === msg.messageId
                      && ttsPlayer.state.status === "playing"
                    }
                    feedbackRating={feedbackRating}
                    feedbackBusy={feedbackBusyMessageId === msg.messageId}
                    sending={sending}
                    canRefresh={
                      msg.role === "assistant"
                      && !!msg.parentUserMessageId
                      && msg.messageId === latestAssistantMessageId
                    }
                    canFeedback={canFeedback}
                    onCopy={() => {
                      void handleCopyMessage(msg);
                    }}
                    onEdit={() => handleEditMessage(msg)}
                    onRefresh={() => handleRefreshMessage(msg)}
                    onListenToggle={() => {
                      void ttsPlayer.toggle(msg.messageId, msg.content);
                    }}
                    onLike={() => {
                      void handleFeedback(msg, "good");
                    }}
                    onDislike={() => {
                      void handleFeedback(msg, "bad");
                    }}
                  />
                ) : null}
                {msg.role === "assistant" && ttsPlayer.state.activeMessageId === msg.messageId ? (
                  <TtsAudioPlayer
                    playback={ttsPlayer.state}
                    onPlayPause={() => {
                      void ttsPlayer.toggle(msg.messageId, msg.content);
                    }}
                    onSeek={ttsPlayer.seek}
                    onStop={ttsPlayer.stop}
                  />
                ) : null}
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {decisionState.shouldRender && decisionState.decision ? (
        <DecisionPanel
          decision={decisionState.decision}
          busy={decisionState.busy}
          error={decisionState.error}
          onRespond={(choice, editedAction) => {
            if (!onDecisionRespond) {
              return;
            }
            void onDecisionRespond(choice, editedAction);
          }}
        />
      ) : null}

      {/* Input area */}
      <div className="px-4 py-3">
        <div className="max-w-5xl mx-auto">
          {visibleStatusMessage ? (
            <div className="mb-2 rounded-lg border border-[#1f1f24] bg-[#141418] px-3 py-2 text-[12px] text-[#c0c0c0]">
              {visibleStatusMessage}
            </div>
          ) : null}
          {sttError ? (
            <div className="mb-2 rounded-lg border border-rose-700/40 bg-rose-900/20 px-3 py-2 text-[12px] text-rose-200">
              {sttError}
            </div>
          ) : null}
          {pasteUndo ? (
            <div className="mb-2 flex items-center justify-between gap-2 rounded-lg border border-[#1f1f24] bg-[#141418] px-3 py-2 text-[12px] text-[#c0c0c0]">
              <span>Вставка длинного текста сохранена как файл.</span>
              <button
                type="button"
                onClick={handleUndoPaste}
                className="rounded border border-[#2a2a30] px-2 py-0.5 text-[11px] text-[#ddd] hover:bg-[#1f1f24]"
              >
                Undo
              </button>
            </div>
          ) : null}
          {composerAttachments.length > 0 ? (
            <div className="mb-2 flex flex-wrap gap-2">
              {composerAttachments.map((attachment) => (
                <div
                  key={attachment.id}
                  className="inline-flex items-center gap-2 rounded-md border border-[#2a2a30] bg-[#141418] px-2.5 py-1 text-[12px] text-[#c8c8cc]"
                >
                  <FileText className="h-3.5 w-3.5 text-[#8f8f95]" />
                  <span className="max-w-[260px] truncate">{attachment.name}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveComposerAttachment(attachment.id)}
                    className="rounded p-0.5 text-[#8f8f95] hover:bg-[#1f1f24] hover:text-[#d6d6db]"
                    title="Remove attachment"
                    aria-label="Remove attachment"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            multiple
            accept="image/*,.txt,.md,.markdown,.json,.yaml,.yml,.toml,.csv,.log,.py,.ts,.tsx,.js,.jsx,.css,.scss,.html,.xml,.sh,.bash,.zsh,.ini,.cfg,.conf,.sql,.env"
            onChange={(event) => {
              void handleAttachFiles(event);
            }}
          />
          <div className="flex items-end gap-2 bg-[#141418] rounded-xl border border-[#1f1f24] focus-within:border-[#2a2a30] transition-colors px-4 py-3">
            {/* Composer actions */}
            <div className="relative flex-shrink-0 pb-0.5">
              {actionsMenuOpen ? (
                <div className="absolute bottom-8 left-0 z-20 min-w-[160px] rounded-lg border border-[#24242a] bg-[#101014] py-1 shadow-xl shadow-black/40">
                  <button
                    type="button"
                    onClick={handleOpenFilePicker}
                    className="flex w-full items-center gap-2 px-3 py-2 text-left text-[13px] text-[#d6d6db] transition-colors hover:bg-[#1a1a20]"
                    title="Add file"
                  >
                    <Paperclip className="h-4 w-4 text-[#8f8f95]" />
                    <span>Add file</span>
                  </button>
                  <button
                    type="button"
                    onClick={handleToggleWebSearch}
                    disabled={!webSearchAvailable}
                    className={`flex w-full items-center gap-2 px-3 py-2 text-left text-[13px] transition-colors disabled:cursor-not-allowed disabled:text-[#555] ${
                      webSearchNext && webSearchAvailable
                        ? "bg-[#173124] text-[#9fe3b8]"
                        : "text-[#d6d6db] hover:bg-[#1a1a20]"
                    }`}
                    title="Web search"
                    aria-pressed={webSearchNext && webSearchAvailable}
                  >
                    <Globe2 className="h-4 w-4 text-current" />
                    <span>Web search</span>
                  </button>
                  <button
                    type="button"
                    onClick={handleToggleCanvasNext}
                    className={`flex w-full items-center gap-2 px-3 py-2 text-left text-[13px] transition-colors ${
                      forceCanvasNext
                        ? "bg-[#173124] text-[#9fe3b8]"
                        : "text-[#d6d6db] hover:bg-[#1a1a20]"
                    }`}
                    title="Canvas"
                    aria-pressed={forceCanvasNext}
                  >
                    <PanelRight className="h-4 w-4 text-current" />
                    <span>Canvas</span>
                  </button>
                </div>
              ) : null}
              <button
                type="button"
                onClick={() => setActionsMenuOpen((prev) => !prev)}
                disabled={actionsDisabled}
                className="flex h-6 w-6 items-center justify-center rounded-md text-[#666] transition-colors hover:bg-[#1b1b20] hover:text-[#aaa] cursor-pointer disabled:cursor-not-allowed disabled:text-[#444]"
                title="Actions"
                aria-label="Actions"
                aria-expanded={actionsMenuOpen}
              >
                <Plus className="h-4.5 w-4.5" />
              </button>
            </div>

            {/* Textarea */}
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onSelect={() => {
                const textarea = textareaRef.current;
                if (!textarea) {
                  return;
                }
                caretSelectionRef.current = {
                  start: textarea.selectionStart ?? 0,
                  end: textarea.selectionEnd ?? 0,
                };
              }}
              onClick={() => {
                const textarea = textareaRef.current;
                if (!textarea) {
                  return;
                }
                caretSelectionRef.current = {
                  start: textarea.selectionStart ?? 0,
                  end: textarea.selectionEnd ?? 0,
                };
              }}
              onKeyUp={() => {
                const textarea = textareaRef.current;
                if (!textarea) {
                  return;
                }
                caretSelectionRef.current = {
                  start: textarea.selectionStart ?? 0,
                  end: textarea.selectionEnd ?? 0,
                };
              }}
              onPaste={handlePaste}
              onKeyDown={handleKeyDown}
              placeholder={
                composerBlocked
                  ? "Нужно решить запрос разрешения перед отправкой."
                  : "Type your message... (Shift+Enter for new line)"
              }
              rows={1}
              className="composer-textarea flex-1 bg-transparent text-[14px] text-[#d4d4d8] placeholder-[#555] resize-none outline-none min-h-[24px] max-h-[120px]"
              style={{ lineHeight: "24px" }}
              disabled={sending || isTranscribing || composerBlocked}
              data-scrollbar="always"
            />

            {/* Mic button */}
            <button
              type="button"
              onClick={() => {
                void handleToggleRecording();
              }}
              disabled={!canUseMediaRecorder || sending || isTranscribing || composerBlocked}
              className={`relative transition-colors pb-0.5 cursor-pointer ${
                !canUseMediaRecorder
                  ? "text-[#444]"
                  : isRecording
                    ? "text-rose-300"
                    : isTranscribing
                      ? "text-amber-300"
                      : "text-[#555] hover:text-[#999]"
              }`}
              title={
                !canUseMediaRecorder
                  ? "Microphone unavailable"
                : isRecording
                  ? "Stop recording"
                  : isTranscribing
                    ? "Transcribing..."
                    : "Start recording"
              }
            >
              {isTranscribing ? (
                <LoaderCircle className="w-4.5 h-4.5 animate-spin" />
              ) : (
                <span className="relative inline-flex items-center justify-center">
                  {isRecording ? <span className="stt-mic-recording" aria-hidden="true" /> : null}
                  <Mic className="relative z-10 w-4.5 h-4.5" />
                </span>
              )}
            </button>

            {/* Send button */}
            <button
              onClick={() => {
                if (sending) {
                  void onCancelSend?.();
                  return;
                }
                void handleSend();
              }}
              disabled={
                sending
                  ? cancelling || !onCancelSend
                  : (!inputValue.trim() && composerAttachments.length === 0)
                    || isTranscribing
                    || composerBlocked
              }
              className={`p-1.5 rounded-lg transition-all cursor-pointer ${
                sending
                  ? cancelling
                    ? "bg-[#1b1b20] text-[#555]"
                    : "bg-rose-600 hover:bg-rose-500 text-white"
                  : (inputValue.trim() || composerAttachments.length > 0)
                    && !isTranscribing
                    && !composerBlocked
                    ? "bg-[#6366f1] hover:bg-[#5558e6] text-white"
                    : "bg-[#1b1b20] text-[#555]"
              }`}
              title={sending ? (cancelling ? "Stopping..." : "Stop generation") : "Send"}
              aria-label={sending ? (cancelling ? "Stopping generation" : "Stop generation") : "Send"}
            >
              {sending ? (
                cancelling ? (
                  <LoaderCircle className="w-4 h-4 animate-spin" />
                ) : (
                  <Square className="w-4 h-4 fill-current" />
                )
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>

          <p className="text-[11px] text-[#444] text-center mt-2">
            SlavikAI v1.0 - Python Agent
          </p>
        </div>
      </div>
    </div>
  );
}
