import { useEffect, useState } from "react";
import {
  Search,
  LayoutGrid,
  FolderClosed,
  MessageSquare,
  Calendar,
  Clock,
  Settings,
  Trash2,
  MoreHorizontal,
  Plus,
} from "lucide-react";

import BrainLogo from "../../assets/brain.png";

export interface ChatItem {
  id: string;
  title: string;
  messageCount: number;
  chatMessageCount?: number;
  workspaceMessageCount?: number;
  date: string;
  group: "today" | "yesterday" | "older";
}

export interface FolderItem {
  id: string;
  name: string;
}

interface HistorySidebarProps {
  chats?: ChatItem[];
  folders?: FolderItem[];
  activeChatId?: string | null;
  onNewChat?: () => void;
  onSelectChat?: (id: string) => void;
  onDeleteChat?: (id: string) => void;
  onRenameChat?: (id: string) => void;
  onMoveChatToFolder?: (id: string, folderId: string | null) => void;
  onOpenSearch?: () => void;
  onOpenWorkspace?: () => void;
  onOpenSettings?: () => void;
  onCreateFolder?: () => void;
  className?: string;
  compact?: boolean;
  workspaceActive?: boolean;
}

const defaultChats: ChatItem[] = [];

export function HistorySidebar({
  chats = defaultChats,
  folders = [],
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onRenameChat,
  onMoveChatToFolder,
  onOpenSearch,
  onOpenWorkspace,
  onOpenSettings,
  onCreateFolder,
  className = "",
  compact = false,
  workspaceActive = false,
}: HistorySidebarProps) {
  const [hoveredChat, setHoveredChat] = useState<string | null>(null);
  const [menuChatId, setMenuChatId] = useState<string | null>(null);
  const [folderPickerChatId, setFolderPickerChatId] = useState<string | null>(null);
  const hasChats = chats.length > 0;
  const hasFolders = folders.length > 0;

  useEffect(() => {
    if (!menuChatId && !folderPickerChatId) {
      return;
    }
    const handleClick = () => {
      setMenuChatId(null);
      setFolderPickerChatId(null);
    };
    document.addEventListener("click", handleClick);
    return () => {
      document.removeEventListener("click", handleClick);
    };
  }, [menuChatId, folderPickerChatId]);

  const todayChats = chats.filter((c) => c.group === "today");
  const yesterdayChats = chats.filter((c) => c.group === "yesterday");
  const olderChats = chats.filter((c) => c.group === "older");
  const workspaceButtonLabel = workspaceActive ? "Close Computer" : "Computer";

  if (compact) {
    return (
      <div className={`flex h-full w-[64px] flex-col items-center bg-zinc-950 py-3 ${className}`}>
        <div className="mb-3 flex h-8 w-8 items-center justify-center">
          <img src={BrainLogo} alt="SlavikAI" className="h-7 w-7 object-contain" />
        </div>
        <button
          onClick={onNewChat}
          className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
          title="New Chat"
          aria-label="New Chat"
        >
          <Plus className="h-4 w-4" />
        </button>
        <button
          onClick={onOpenSearch}
          className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg text-zinc-400 hover:bg-zinc-900"
          title="Search"
          aria-label="Search"
        >
          <Search className="h-4 w-4" />
        </button>
        <button
          onClick={onOpenWorkspace}
          className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
          title={workspaceButtonLabel}
          aria-label={workspaceButtonLabel}
          aria-pressed={workspaceActive}
        >
          <LayoutGrid className="h-4 w-4" />
        </button>
        <div className="mt-auto">
          <button
            onClick={onOpenSettings}
            className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-400 hover:bg-zinc-900"
            title="Settings"
            aria-label="Settings"
          >
            <Settings className="h-4 w-4" />
          </button>
        </div>
      </div>
    );
  }

  const formatDate = (dateStr: string) => {
    try {
      const d = new Date(dateStr);
      return d.toLocaleString("en-US", {
        month: "numeric",
        day: "numeric",
        year: "numeric",
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
      });
    } catch {
      return dateStr;
    }
  };

  const ChatGroup = ({
    label,
    icon,
    items,
  }: {
    label: string;
    icon: React.ReactNode;
    items: ChatItem[];
  }) => {
    if (items.length === 0) return null;
    return (
      <div className="mb-3">
        <div className="flex items-center gap-2 px-3 py-1.5">
          {icon}
          <span className="text-[11px] text-zinc-500 uppercase tracking-wider">
            {label}
          </span>
        </div>
        {items.map((chat) => (
          <div
            key={chat.id}
            className={`group relative flex items-center gap-2 mx-2 px-3 py-2.5 rounded-lg cursor-pointer transition-all ${
              activeChatId === chat.id
                ? "bg-zinc-800"
                : "hover:bg-zinc-900"
            }`}
            onClick={() => {
              onSelectChat?.(chat.id);
              setMenuChatId(null);
              setFolderPickerChatId(null);
            }}
            onMouseEnter={() => setHoveredChat(chat.id)}
            onMouseLeave={() => setHoveredChat(null)}
          >
            <div className="flex-1 min-w-0">
              <div className="text-[13px] text-zinc-300 truncate">
                {chat.title}
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-[11px] text-zinc-500">
                  C:{chat.chatMessageCount ?? chat.messageCount}
                </span>
                <span className="text-[11px] text-zinc-500">
                  Tools:{chat.workspaceMessageCount ?? 0}
                </span>
                <span className="text-[11px] text-zinc-600">-</span>
                <span className="text-[11px] text-zinc-500">
                  {formatDate(chat.date)}
                </span>
              </div>
            </div>

            {/* Action buttons on hover */}
            <div
              className={`flex items-center gap-1 flex-shrink-0 transition-opacity ${
                hoveredChat === chat.id ? "opacity-100" : "opacity-0"
              }`}
            >
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteChat?.(chat.id);
                  setMenuChatId(null);
                  setFolderPickerChatId(null);
                }}
                className="p-1 rounded text-zinc-500 hover:text-red-400 hover:bg-red-400/10 transition-colors cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setFolderPickerChatId(null);
                  setMenuChatId((prev) => (prev === chat.id ? null : chat.id));
                }}
                className="p-1 rounded text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700 transition-colors cursor-pointer"
              >
                <MoreHorizontal className="w-3.5 h-3.5" />
              </button>
            </div>

            {menuChatId === chat.id ? (
              <div
                className="absolute right-2 top-10 z-20 w-44 rounded-lg bg-zinc-900 border border-zinc-800 shadow-xl shadow-black/40 py-1"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => {
                    onRenameChat?.(chat.id);
                    setMenuChatId(null);
                    setFolderPickerChatId(null);
                  }}
                  className="w-full px-3 py-2 text-left text-[12px] text-zinc-200 hover:bg-zinc-800 transition-colors"
                >
                  Rename
                </button>
                <button
                  onClick={() => {
                    setMenuChatId(null);
                    setFolderPickerChatId(chat.id);
                  }}
                  className="w-full px-3 py-2 text-left text-[12px] text-zinc-200 hover:bg-zinc-800 transition-colors"
                >
                  Send to folder
                </button>
              </div>
            ) : null}

            {folderPickerChatId === chat.id ? (
              <div
                className="absolute right-2 top-10 z-20 w-48 rounded-lg bg-zinc-900 border border-zinc-800 shadow-xl shadow-black/40 py-1"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => {
                    onMoveChatToFolder?.(chat.id, null);
                    setFolderPickerChatId(null);
                  }}
                  className="w-full px-3 py-2 text-left text-[12px] text-zinc-200 hover:bg-zinc-800 transition-colors"
                >
                  No folder
                </button>
                {hasFolders ? (
                  folders.map((folder) => (
                    <button
                      key={folder.id}
                      onClick={() => {
                        onMoveChatToFolder?.(chat.id, folder.id);
                        setFolderPickerChatId(null);
                      }}
                      className="w-full px-3 py-2 text-left text-[12px] text-zinc-200 hover:bg-zinc-800 transition-colors"
                    >
                      {folder.name}
                    </button>
                  ))
                ) : (
                  <button
                    onClick={() => {
                      onCreateFolder?.();
                      setFolderPickerChatId(null);
                    }}
                    className="w-full px-3 py-2 text-left text-[12px] text-zinc-400 hover:bg-zinc-800 transition-colors"
                  >
                    Create folder
                  </button>
                )}
              </div>
            ) : null}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div
      className={`flex flex-col h-full w-[260px] bg-zinc-950 ${className}`}
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-5 py-4">
        <div className="w-7 h-7 flex items-center justify-center">
          <img
            src={BrainLogo}
            alt="SlavikAI"
            className="w-7 h-7 object-contain"
          />
        </div>
        <span className="text-[15px] text-zinc-200">SlavikAI</span>
      </div>

      {/* New Chat button */}
      <div className="px-3 mb-3">
        <button
          onClick={onNewChat}
          className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-[13px] text-zinc-300 transition-all cursor-pointer"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {/* Quick actions */}
      <div className="px-2 mb-2 space-y-0.5">
        <button
          onClick={onOpenSearch}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-all cursor-pointer"
        >
          <Search className="w-4 h-4" />
          Search
        </button>
        <button
          onClick={onOpenWorkspace}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-all cursor-pointer"
        >
          <LayoutGrid className="w-4 h-4" />
          Computer
        </button>
      </div>

      {/* Folders section */}
      <div className="px-3 mb-2">
        <div className="flex items-center justify-between px-2 py-1.5">
          <div className="flex items-center gap-2">
            <FolderClosed className="w-3.5 h-3.5 text-zinc-500" />
            <span className="text-[11px] text-zinc-500 uppercase tracking-wider">
              Folders
            </span>
          </div>
          <button
            onClick={onCreateFolder}
            className="p-1 rounded text-zinc-500 hover:text-zinc-200 hover:bg-zinc-900 transition-colors cursor-pointer"
            title="Create folder"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
        </div>
        {hasFolders ? (
          <div className="space-y-1 px-1 pb-2">
            {folders.map((folder) => (
              <div
                key={folder.id}
                className="mx-2 rounded-lg px-2 py-1.5 text-[12px] text-zinc-400 hover:bg-zinc-900 transition-colors"
              >
                {folder.name}
              </div>
            ))}
          </div>
        ) : (
          <p className="px-3 py-2 text-[12px] text-zinc-600 italic">
            No folders yet
          </p>
        )}
      </div>

      {/* Divider */}
      {/* Chats */}
      <div className="px-1 mb-1">
        <div className="flex items-center gap-2 px-4 py-1.5">
          <MessageSquare className="w-3.5 h-3.5 text-zinc-500" />
          <span className="text-[11px] text-zinc-500 uppercase tracking-wider">
            Chats
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto" data-scrollbar="always">
        {hasChats ? (
          <>
            <ChatGroup
              label="Today"
              icon={<Calendar className="w-3 h-3 text-zinc-500" />}
              items={todayChats}
            />
            <ChatGroup
              label="Yesterday"
              icon={<Clock className="w-3 h-3 text-zinc-500" />}
              items={yesterdayChats}
            />
            <ChatGroup
              label="Older"
              icon={<Clock className="w-3 h-3 text-zinc-500" />}
              items={olderChats}
            />
          </>
        ) : (
          <div className="px-4 py-6 text-[12px] text-zinc-500">
            No chats yet
          </div>
        )}
      </div>

      {/* Bottom section */}
      <div className="px-3 py-3 space-y-2">
        <button
          onClick={onOpenSettings}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-all cursor-pointer"
        >
          <Settings className="w-4 h-4" />
          Settings
        </button>
        <p className="text-[11px] text-zinc-600 text-center">
          SlavikAI v1.0
        </p>
      </div>
    </div>
  );
}
