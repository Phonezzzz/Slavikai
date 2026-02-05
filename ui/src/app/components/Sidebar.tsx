import {
  ChevronLeft,
  ChevronRight,
  MessageSquarePlus,
  Settings as SettingsIcon,
  Sparkles,
  Trash2,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';

import type { SessionSummary } from '../types';

type SidebarProps = {
  selectedConversation: string | null;
  conversations: SessionSummary[];
  loading: boolean;
  deletingSessionId: string | null;
  onSelectConversation: (id: string) => void;
  onCreateConversation: () => void;
  onDeleteConversation: (id: string) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onOpenSettings: () => void;
};

type SessionGroup = {
  key: string;
  label: string;
  sessions: SessionSummary[];
};

const formatUpdatedAt = (updatedAt: string): string => {
  const parsed = Date.parse(updatedAt);
  if (Number.isNaN(parsed)) {
    return updatedAt;
  }
  return new Date(parsed).toLocaleString();
};

const groupSessions = (sessions: SessionSummary[]): SessionGroup[] => {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfSessionDay = (value: string): Date | null => {
    const parsed = Date.parse(value);
    if (Number.isNaN(parsed)) {
      return null;
    }
    const date = new Date(parsed);
    return new Date(date.getFullYear(), date.getMonth(), date.getDate());
  };
  const bucketFor = (value: string): string => {
    const day = startOfSessionDay(value);
    if (!day) {
      return 'older';
    }
    const diffDays = Math.floor((startOfToday.getTime() - day.getTime()) / 86_400_000);
    if (diffDays <= 0) {
      return 'today';
    }
    if (diffDays === 1) {
      return 'yesterday';
    }
    if (diffDays <= 7) {
      return 'last7';
    }
    return 'older';
  };
  const buckets: Record<string, SessionSummary[]> = {
    today: [],
    yesterday: [],
    last7: [],
    older: [],
  };
  sessions.forEach((session) => {
    buckets[bucketFor(session.updated_at)].push(session);
  });
  return [
    { key: 'today', label: '📅 Сегодня', sessions: buckets.today },
    { key: 'yesterday', label: '🕒 Вчера', sessions: buckets.yesterday },
    { key: 'last7', label: '🗂️ Предыдущие 7 дней', sessions: buckets.last7 },
    { key: 'older', label: '🧾 Ранее', sessions: buckets.older },
  ].filter((group) => group.sessions.length > 0);
};

export function Sidebar({
  selectedConversation,
  conversations,
  loading,
  deletingSessionId,
  onSelectConversation,
  onCreateConversation,
  onDeleteConversation,
  collapsed,
  onToggleCollapse,
  onOpenSettings,
}: SidebarProps) {
  return (
    <motion.div
      initial={false}
      animate={{ width: collapsed ? '24px' : '300px' }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="relative bg-zinc-950"
    >
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex h-full w-[300px] flex-col"
          >
            <div className="p-4">
              <div className="mb-4 flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/20 bg-white/10">
                  <Sparkles className="h-5 w-5 text-white" />
                </div>
                <h1 className="text-lg font-semibold text-white">SlavikAI</h1>
              </div>

              <button
                type="button"
                onClick={onCreateConversation}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-white px-4 py-2.5 font-medium text-black transition-all duration-200 hover:bg-white/90"
              >
                <MessageSquarePlus className="h-4 w-4" />
                New Chat
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-2">
              <div className="mb-2 px-3 text-[11px] font-semibold uppercase tracking-wide text-white/40">
                📁 Папки
              </div>
              <div className="mb-4 rounded-lg border border-white/5 bg-white/5 px-3 py-2 text-xs text-white/40">
                Папок пока нет
              </div>
              <div className="mb-2 px-3 text-[11px] font-semibold uppercase tracking-wide text-white/40">
                💬 Чаты
              </div>
              {loading ? (
                <div className="rounded-lg px-3 py-2.5 text-sm text-white/50">Loading...</div>
              ) : conversations.length === 0 ? (
                <div className="rounded-lg px-3 py-2.5 text-sm text-white/50">No chats yet</div>
              ) : (
                <div className="space-y-3">
                  {groupSessions(
                    [...conversations].sort(
                      (a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at),
                    ),
                  ).map((group) => (
                    <div key={group.key} className="space-y-1">
                      <div className="px-3 py-1 text-xs font-medium text-white/40">{group.label}</div>
                      {group.sessions.map((conv) => {
                        const isActive = selectedConversation === conv.session_id;
                        const isDeleting = deletingSessionId === conv.session_id;
                        return (
                          <button
                            key={conv.session_id}
                            type="button"
                            onClick={() => onSelectConversation(conv.session_id)}
                            className={`group w-full rounded-lg px-3 py-2.5 text-left transition-all duration-200 ${
                              isActive
                                ? 'bg-white/10 text-white shadow-lg'
                                : 'text-white/60 hover:bg-white/5 hover:text-white/90'
                            }`}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <div className="min-w-0">
                                <div className="truncate text-sm font-medium">{conv.title}</div>
                                <div className="truncate text-xs text-white/40">
                                  {conv.message_count} messages
                                </div>
                                <div className="truncate text-xs text-white/35">
                                  {conv.session_id.slice(0, 8)}
                                </div>
                                <div className="mt-1 truncate text-xs text-white/30">
                                  {formatUpdatedAt(conv.updated_at)}
                                </div>
                              </div>
                              <span
                                role="button"
                                tabIndex={0}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  if (!isDeleting) {
                                    onDeleteConversation(conv.session_id);
                                  }
                                }}
                                onKeyDown={(event) => {
                                  if (event.key === 'Enter' || event.key === ' ') {
                                    event.preventDefault();
                                    event.stopPropagation();
                                    if (!isDeleting) {
                                      onDeleteConversation(conv.session_id);
                                    }
                                  }
                                }}
                                className="rounded-md p-1 text-white/40 transition-colors hover:bg-white/10 hover:text-white/80"
                                aria-label={`Delete chat ${conv.session_id}`}
                              >
                                {isDeleting ? (
                                  <span className="px-1 text-[10px] uppercase tracking-wide">...</span>
                                ) : (
                                  <Trash2 className="h-3.5 w-3.5" />
                                )}
                              </span>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-3 p-4">
              <button
                type="button"
                onClick={onOpenSettings}
                className="flex w-full items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/5 px-4 py-2.5 font-medium text-white/70 transition-all duration-200 hover:bg-white/10 hover:text-white"
              >
                <SettingsIcon className="h-4 w-4" />
                Settings
              </button>
              <div className="text-center text-xs text-white/40">SlavikAI v1.0 • Python Agent</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <button
        type="button"
        onClick={onToggleCollapse}
        className="absolute -right-3 top-1/2 z-20 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full border border-white/10 bg-white/10 backdrop-blur-xl transition-all duration-200 hover:bg-white/20"
      >
        {collapsed ? (
          <ChevronRight className="h-3.5 w-3.5 text-white/60" />
        ) : (
          <ChevronLeft className="h-3.5 w-3.5 text-white/60" />
        )}
      </button>
    </motion.div>
  );
}
