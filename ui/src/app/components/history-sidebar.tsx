import { useEffect, useState } from "react";
import {
  Search,
  StickyNote,
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
  onOpenNotes?: () => void;
  onOpenWorkspace?: () => void;
  onOpenSettings?: () => void;
  onCreateFolder?: () => void;
  className?: string;
  compact?: boolean;
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
  onOpenNotes,
  onOpenWorkspace,
  onOpenSettings,
  onCreateFolder,
  className = "",
  compact = false,
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

  if (compact) {
    return (
      <div className={`flex h-full w-[64px] flex-col items-center bg-[#0b0b0d] py-3 ${className}`}>
        <div className="mb-3 flex h-8 w-8 items-center justify-center">
          <img src={BrainLogo} alt="SlavikAI" className="h-7 w-7 object-contain" />
        </div>
        <button
          onClick={onNewChat}
          className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-[#141418] text-[#d0d0d0] hover:bg-[#1b1b20]"
          title="New Chat"
          aria-label="New Chat"
        >
          <Plus className="h-4 w-4" />
        </button>
        <button
          onClick={onOpenSearch}
          className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg text-[#b3b3bb] hover:bg-[#15151a]"
          title="Search"
          aria-label="Search"
        >
          <Search className="h-4 w-4" />
        </button>
        <button
          onClick={onOpenNotes}
          className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg text-[#b3b3bb] hover:bg-[#15151a]"
          title="Notes"
          aria-label="Notes"
        >
          <StickyNote className="h-4 w-4" />
        </button>
        <button
          onClick={onOpenWorkspace}
          className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-[#141418] text-[#d0d0d0] hover:bg-[#1b1b20]"
          title="Workspace"
          aria-label="Workspace"
        >
          <LayoutGrid className="h-4 w-4" />
        </button>
        <div className="mt-auto">
          <button
            onClick={onOpenSettings}
            className="flex h-9 w-9 items-center justify-center rounded-lg text-[#b3b3bb] hover:bg-[#15151a]"
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
          <span className="text-[11px] text-[#666] uppercase tracking-wider">
            {label}
          </span>
        </div>
        {items.map((chat) => (
          <div
            key={chat.id}
            className={`group relative flex items-center gap-2 mx-2 px-3 py-2.5 rounded-lg cursor-pointer transition-all ${
              activeChatId === chat.id
                ? "bg-[#1b1b22]"
                : "hover:bg-[#141418]"
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
              <div className="text-[13px] text-[#d0d0d0] truncate">
                {chat.title}
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-[11px] text-[#666]">
                  {chat.messageCount} messages
                </span>
                <span className="text-[11px] text-[#444]">-</span>
                <span className="text-[11px] text-[#555]">
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
                className="p-1 rounded text-[#555] hover:text-red-400 hover:bg-red-400/10 transition-colors cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setFolderPickerChatId(null);
                  setMenuChatId((prev) => (prev === chat.id ? null : chat.id));
                }}
                className="p-1 rounded text-[#555] hover:text-[#aaa] hover:bg-[#333] transition-colors cursor-pointer"
              >
                <MoreHorizontal className="w-3.5 h-3.5" />
              </button>
            </div>

            {menuChatId === chat.id ? (
              <div
                className="absolute right-2 top-10 z-20 w-44 rounded-lg bg-[#141418] border border-[#1f1f24] shadow-xl shadow-black/40 py-1"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => {
                    onRenameChat?.(chat.id);
                    setMenuChatId(null);
                    setFolderPickerChatId(null);
                  }}
                  className="w-full px-3 py-2 text-left text-[12px] text-[#ddd] hover:bg-[#1b1b20] transition-colors"
                >
                  Rename
                </button>
                <button
                  onClick={() => {
                    setMenuChatId(null);
                    setFolderPickerChatId(chat.id);
                  }}
                  className="w-full px-3 py-2 text-left text-[12px] text-[#ddd] hover:bg-[#1b1b20] transition-colors"
                >
                  Send to folder
                </button>
              </div>
            ) : null}

            {folderPickerChatId === chat.id ? (
              <div
                className="absolute right-2 top-10 z-20 w-48 rounded-lg bg-[#141418] border border-[#1f1f24] shadow-xl shadow-black/40 py-1"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => {
                    onMoveChatToFolder?.(chat.id, null);
                    setFolderPickerChatId(null);
                  }}
                  className="w-full px-3 py-2 text-left text-[12px] text-[#ddd] hover:bg-[#1b1b20] transition-colors"
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
                      className="w-full px-3 py-2 text-left text-[12px] text-[#ddd] hover:bg-[#1b1b20] transition-colors"
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
                    className="w-full px-3 py-2 text-left text-[12px] text-[#999] hover:bg-[#1b1b20] transition-colors"
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
      className={`flex flex-col h-full w-[260px] bg-[#0b0b0d] ${className}`}
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
        <span className="text-[15px] text-[#e0e0e0]">SlavikAI</span>
      </div>

      {/* New Chat button */}
      <div className="px-3 mb-3">
        <button
          onClick={onNewChat}
          className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl bg-[#141418] hover:bg-[#1b1b20] text-[13px] text-[#d0d0d0] transition-all cursor-pointer"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {/* Quick actions */}
      <div className="px-2 mb-2 space-y-0.5">
        <button
          onClick={onOpenSearch}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-[#999] hover:text-[#ddd] hover:bg-[#141418] transition-all cursor-pointer"
        >
          <Search className="w-4 h-4" />
          Search
        </button>
        <button
          onClick={onOpenNotes}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-[#999] hover:text-[#ddd] hover:bg-[#141418] transition-all cursor-pointer"
        >
          <StickyNote className="w-4 h-4" />
          Notes
        </button>
        <button
          onClick={onOpenWorkspace}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-[#999] hover:text-[#ddd] hover:bg-[#141418] transition-all cursor-pointer"
        >
          <LayoutGrid className="w-4 h-4" />
          Workspace
        </button>
      </div>

      {/* Folders section */}
      <div className="px-3 mb-2">
        <div className="flex items-center justify-between px-2 py-1.5">
          <div className="flex items-center gap-2">
            <FolderClosed className="w-3.5 h-3.5 text-[#555]" />
            <span className="text-[11px] text-[#555] uppercase tracking-wider">
              Folders
            </span>
          </div>
          <button
            onClick={onCreateFolder}
            className="p-1 rounded text-[#555] hover:text-[#ddd] hover:bg-[#141418] transition-colors cursor-pointer"
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
                className="mx-2 rounded-lg px-2 py-1.5 text-[12px] text-[#bbb] hover:bg-[#141418] transition-colors"
              >
                {folder.name}
              </div>
            ))}
          </div>
        ) : (
          <p className="px-3 py-2 text-[12px] text-[#444] italic">
            No folders yet
          </p>
        )}
      </div>

      {/* Divider */}
      {/* Chats */}
      <div className="px-1 mb-1">
        <div className="flex items-center gap-2 px-4 py-1.5">
          <MessageSquare className="w-3.5 h-3.5 text-[#555]" />
          <span className="text-[11px] text-[#555] uppercase tracking-wider">
            Chats
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto" data-scrollbar="always">
        {hasChats ? (
          <>
            <ChatGroup
              label="Today"
              icon={<Calendar className="w-3 h-3 text-[#555]" />}
              items={todayChats}
            />
            <ChatGroup
              label="Yesterday"
              icon={<Clock className="w-3 h-3 text-[#555]" />}
              items={yesterdayChats}
            />
            <ChatGroup
              label="Older"
              icon={<Clock className="w-3 h-3 text-[#555]" />}
              items={olderChats}
            />
          </>
        ) : (
          <div className="px-4 py-6 text-[12px] text-[#555]">
            No chats yet
          </div>
        )}
      </div>

      {/* Bottom section */}
      <div className="px-3 py-3 space-y-2">
        <button
          onClick={onOpenSettings}
          className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-[13px] text-[#999] hover:text-[#ddd] hover:bg-[#141418] transition-all cursor-pointer"
        >
          <Settings className="w-4 h-4" />
          Settings
        </button>
        <p className="text-[11px] text-[#444] text-center">
          SlavikAI v1.0
        </p>
      </div>
    </div>
  );
}
