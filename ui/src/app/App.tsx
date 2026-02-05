import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { Workspace } from './components/Workspace';
import { Settings } from './components/Settings';

export default function App() {
  const [selectedConversation, setSelectedConversation] = useState<string | null>('conv-1');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [workspaceCollapsed, setWorkspaceCollapsed] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <div className="flex h-screen bg-zinc-950 text-foreground overflow-hidden">
      {/* Left Sidebar */}
      <Sidebar
        selectedConversation={selectedConversation}
        onSelectConversation={setSelectedConversation}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      {/* Main Chat Area */}
      <ChatArea conversationId={selectedConversation} />

      {/* Workspace Panel */}
      <Workspace
        collapsed={workspaceCollapsed}
        onToggleCollapse={() => setWorkspaceCollapsed(!workspaceCollapsed)}
      />

      {/* Settings Modal */}
      <Settings
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
