import { X, Key, Sliders, MessageSquare, Upload, Download } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useState } from 'react';

interface SettingsProps {
  isOpen: boolean;
  onClose: () => void;
}

type Provider = 'xai' | 'openrouter' | 'local';

export function Settings({ isOpen, onClose }: SettingsProps) {
  const [activeTab, setActiveTab] = useState<'api' | 'personalization' | 'memory' | 'tools' | 'import'>('api');
  const [selectedProvider, setSelectedProvider] = useState<Provider>('local');
  const [apiKeys, setApiKeys] = useState({
    xai: '',
    openrouter: '',
    local: '',
  });
  const [systemPrompt, setSystemPrompt] = useState('You are SlavikAI, a helpful AI assistant with MWV architecture.');
  const [tone, setTone] = useState('professional');

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50"
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="bg-zinc-900 border border-white/10 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-white/10">
                <div>
                  <div className="text-xs text-white/40 mb-1">SETTINGS</div>
                  <h2 className="text-xl font-semibold text-white">Workspace controls</h2>
                </div>
                <div className="flex gap-2">
                  <button className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-sm text-white/70 hover:text-white transition-all">
                    Refresh
                  </button>
                  <button
                    onClick={onClose}
                    className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-sm text-white/70 hover:text-white transition-all"
                  >
                    Close
                  </button>
                </div>
              </div>

              {/* Content */}
              <div className="flex flex-1 overflow-hidden">
                {/* Sidebar */}
                <div className="w-56 border-r border-white/10 p-4 space-y-1">
                  <button
                    onClick={() => setActiveTab('api')}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all ${
                      activeTab === 'api' ? 'bg-white text-black' : 'text-white/70 hover:bg-white/5'
                    }`}
                  >
                    API Keys / Providers
                  </button>
                  <button
                    onClick={() => setActiveTab('personalization')}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all ${
                      activeTab === 'personalization' ? 'bg-white text-black' : 'text-white/70 hover:bg-white/5'
                    }`}
                  >
                    Personalization
                  </button>
                  <button
                    onClick={() => setActiveTab('memory')}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all ${
                      activeTab === 'memory' ? 'bg-white text-black' : 'text-white/70 hover:bg-white/5'
                    }`}
                  >
                    Memory
                  </button>
                  <button
                    onClick={() => setActiveTab('tools')}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all ${
                      activeTab === 'tools' ? 'bg-white text-black' : 'text-white/70 hover:bg-white/5'
                    }`}
                  >
                    Tools
                  </button>
                  <button
                    onClick={() => setActiveTab('import')}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all ${
                      activeTab === 'import' ? 'bg-white text-black' : 'text-white/70 hover:bg-white/5'
                    }`}
                  >
                    Import / Export chats DB
                  </button>
                </div>

                {/* Main Content */}
                <div className="flex-1 p-6 overflow-y-auto">
                  {activeTab === 'api' && (
                    <div className="space-y-6">
                      {/* XAI Provider */}
                      <div className="bg-black/40 border border-white/10 rounded-xl p-4">
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="font-medium text-white">xai</h3>
                          <span className="px-2 py-1 rounded-md bg-yellow-500/20 text-yellow-400 text-xs font-medium">
                            key missing
                          </span>
                        </div>
                        <div className="space-y-2 text-sm">
                          <div>
                            <span className="text-white/50">Env: </span>
                            <span className="text-white/90 font-mono">XAI_API_KEY</span>
                          </div>
                          <div>
                            <span className="text-white/50">Endpoint: </span>
                            <span className="text-white/70">https://api.x.ai/v1/models</span>
                          </div>
                          <div>
                            <span className="text-white/50">Available models: </span>
                            <span className="text-white/70">0</span>
                          </div>
                          <div className="text-red-400 text-xs">Не задан XAI_API_KEY.</div>
                          <input
                            type="password"
                            placeholder="Enter XAI API Key"
                            value={apiKeys.xai}
                            onChange={(e) => setApiKeys({ ...apiKeys, xai: e.target.value })}
                            className="w-full mt-2 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-white/20"
                          />
                        </div>
                      </div>

                      {/* OpenRouter Provider */}
                      <div className="bg-black/40 border border-white/10 rounded-xl p-4">
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="font-medium text-white">openrouter</h3>
                          <span className="px-2 py-1 rounded-md bg-yellow-500/20 text-yellow-400 text-xs font-medium">
                            key missing
                          </span>
                        </div>
                        <div className="space-y-2 text-sm">
                          <div>
                            <span className="text-white/50">Env: </span>
                            <span className="text-white/90 font-mono">OPENROUTER_API_KEY</span>
                          </div>
                          <div>
                            <span className="text-white/50">Endpoint: </span>
                            <span className="text-white/70">https://openrouter.ai/api/v1/models</span>
                          </div>
                          <div>
                            <span className="text-white/50">Available models: </span>
                            <span className="text-white/70">0</span>
                          </div>
                          <div className="text-red-400 text-xs">Не задан OPENROUTER_API_KEY.</div>
                          <input
                            type="password"
                            placeholder="Enter OpenRouter API Key"
                            value={apiKeys.openrouter}
                            onChange={(e) => setApiKeys({ ...apiKeys, openrouter: e.target.value })}
                            className="w-full mt-2 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-white/20"
                          />
                        </div>
                      </div>

                      {/* Local Provider */}
                      <div className="bg-black/40 border border-white/10 rounded-xl p-4">
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="font-medium text-white">local</h3>
                          <span className="px-2 py-1 rounded-md bg-yellow-500/20 text-yellow-400 text-xs font-medium">
                            key missing
                          </span>
                        </div>
                        <div className="space-y-2 text-sm">
                          <div>
                            <span className="text-white/50">Env: </span>
                            <span className="text-white/90 font-mono">LOCAL_LLM_API_KEY</span>
                          </div>
                          <div>
                            <span className="text-white/50">Endpoint: </span>
                            <span className="text-white/70">http://localhost:11434/v1/chat/completions</span>
                          </div>
                          <div>
                            <span className="text-white/50">Available models: </span>
                            <span className="text-white/70">11</span>
                          </div>
                          <input
                            type="text"
                            placeholder="Enter Local API Key (optional)"
                            value={apiKeys.local}
                            onChange={(e) => setApiKeys({ ...apiKeys, local: e.target.value })}
                            className="w-full mt-2 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-white/20"
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'personalization' && (
                    <div className="space-y-6">
                      <div>
                        <label className="block text-sm font-medium text-white/70 mb-2">System Prompt</label>
                        <textarea
                          value={systemPrompt}
                          onChange={(e) => setSystemPrompt(e.target.value)}
                          rows={6}
                          className="w-full px-4 py-3 bg-black/40 border border-white/10 rounded-xl text-white text-sm placeholder-white/30 resize-none focus:outline-none focus:ring-2 focus:ring-white/20"
                          placeholder="Enter system prompt..."
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-white/70 mb-2">Tone</label>
                        <select
                          value={tone}
                          onChange={(e) => setTone(e.target.value)}
                          className="w-full px-4 py-3 bg-black/40 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:ring-2 focus:ring-white/20"
                        >
                          <option value="professional">Professional</option>
                          <option value="casual">Casual</option>
                          <option value="technical">Technical</option>
                          <option value="friendly">Friendly</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-white/70 mb-2">Model Provider</label>
                        <div className="grid grid-cols-3 gap-2">
                          <button
                            onClick={() => setSelectedProvider('local')}
                            className={`px-4 py-3 rounded-lg border transition-all ${
                              selectedProvider === 'local'
                                ? 'bg-white text-black border-white'
                                : 'bg-black/40 text-white/70 border-white/10 hover:bg-white/5'
                            }`}
                          >
                            Local
                          </button>
                          <button
                            onClick={() => setSelectedProvider('openrouter')}
                            className={`px-4 py-3 rounded-lg border transition-all ${
                              selectedProvider === 'openrouter'
                                ? 'bg-white text-black border-white'
                                : 'bg-black/40 text-white/70 border-white/10 hover:bg-white/5'
                            }`}
                          >
                            OpenRouter
                          </button>
                          <button
                            onClick={() => setSelectedProvider('xai')}
                            className={`px-4 py-3 rounded-lg border transition-all ${
                              selectedProvider === 'xai'
                                ? 'bg-white text-black border-white'
                                : 'bg-black/40 text-white/70 border-white/10 hover:bg-white/5'
                            }`}
                          >
                            XAI
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'memory' && (
                    <div className="space-y-4">
                      <p className="text-white/60 text-sm">Memory settings coming soon...</p>
                    </div>
                  )}

                  {activeTab === 'tools' && (
                    <div className="space-y-4">
                      <p className="text-white/60 text-sm">Tool configuration coming soon...</p>
                    </div>
                  )}

                  {activeTab === 'import' && (
                    <div className="space-y-4">
                      <button className="w-full px-4 py-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white/70 hover:text-white transition-all flex items-center justify-center gap-2">
                        <Upload className="w-4 h-4" />
                        Import Chats Database
                      </button>
                      <button className="w-full px-4 py-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white/70 hover:text-white transition-all flex items-center justify-center gap-2">
                        <Download className="w-4 h-4" />
                        Export Chats Database
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
