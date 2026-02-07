import { useState } from "react";
import {
  FileText,
  Download,
  Code,
  Play,
  X,
  ChevronRight,
  File,
} from "lucide-react";

export interface Artifact {
  id: string;
  name: string;
  type: "MD" | "PY" | "JS" | "TS" | "JSON" | "TXT" | "HTML" | "CSS";
  category: "Document" | "Code" | "Config" | "Script";
  content?: string;
}

export interface ContentItem {
  id: string;
  thumbnail?: string;
  title?: string;
  isVideo?: boolean;
}

interface ArtifactsSidebarProps {
  artifacts?: Artifact[];
  contentItems?: ContentItem[];
  onDownloadAll?: () => void;
  onDownloadArtifact?: (id: string) => void;
  onArtifactClick?: (artifact: Artifact) => void;
  onContentClick?: (id: string) => void;
  onClose?: () => void;
  className?: string;
}

const defaultArtifacts: Artifact[] = [
  { id: "1", name: "Plan fixed", type: "MD", category: "Document" },
  { id: "2", name: "Plan", type: "MD", category: "Document" },
  { id: "3", name: "Hello", type: "PY", category: "Script" },
];

const defaultContentItems: ContentItem[] = [
  { id: "c1", title: "Terminal output 1" },
  { id: "c2", title: "Terminal output 2" },
  { id: "c3", title: "Terminal output 3", isVideo: true },
];

const typeColorMap: Record<string, string> = {
  MD: "text-blue-400",
  PY: "text-yellow-400",
  JS: "text-yellow-300",
  TS: "text-blue-300",
  JSON: "text-green-400",
  TXT: "text-gray-400",
  HTML: "text-orange-400",
  CSS: "text-purple-400",
};

const categoryIconMap: Record<string, React.ReactNode> = {
  Document: <FileText className="w-5 h-5 text-gray-400" />,
  Code: <Code className="w-5 h-5 text-gray-400" />,
  Config: <File className="w-5 h-5 text-gray-400" />,
  Script: <Code className="w-5 h-5 text-gray-400" />,
};

export function ArtifactsSidebar({
  artifacts = defaultArtifacts,
  contentItems = defaultContentItems,
  onDownloadAll,
  onDownloadArtifact,
  onArtifactClick,
  onContentClick,
  onClose,
  className = "",
}: ArtifactsSidebarProps) {
  const [hoveredArtifact, setHoveredArtifact] = useState<string | null>(null);

  return (
    <div
      className={`flex flex-col h-full w-[340px] bg-[#1a1a1e] border-l border-[#2a2a2e] ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-[#2a2a2e]">
        <h3 className="text-[#e0e0e0] text-[15px]">Artifacts</h3>
        <div className="flex items-center gap-3">
          <button
            onClick={onDownloadAll}
            className="flex items-center gap-1.5 text-[13px] text-[#888] hover:text-[#ccc] transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            Download all
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="text-[#666] hover:text-[#ccc] transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto" data-scrollbar>
        <div className="p-4 space-y-2">
          {/* Artifacts List */}
          {artifacts.length === 0 ? (
            <div className="rounded-xl border border-[#2a2a2e] bg-[#141418] p-4 text-[13px] text-[#777]">
              Нет артефактов.
            </div>
          ) : (
            artifacts.map((artifact) => (
            <div
              key={artifact.id}
              className="group flex items-center gap-3 p-3 rounded-xl bg-[#222226] hover:bg-[#2a2a30] border border-[#2f2f35] hover:border-[#3a3a42] transition-all cursor-pointer"
              onMouseEnter={() => setHoveredArtifact(artifact.id)}
              onMouseLeave={() => setHoveredArtifact(null)}
              onClick={() => onArtifactClick?.(artifact)}
            >
              <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-[#2a2a30] border border-[#333338]">
                {categoryIconMap[artifact.category]}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[14px] text-[#e0e0e0] truncate">
                  {artifact.name}
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="text-[12px] text-[#777]">
                    {artifact.category}
                  </span>
                  <span className="text-[12px] text-[#555]">-</span>
                  <span
                    className={`text-[12px] ${typeColorMap[artifact.type] || "text-gray-400"}`}
                  >
                    {artifact.type}
                  </span>
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDownloadArtifact?.(artifact.id);
                }}
                className={`text-[#555] hover:text-[#ccc] transition-all ${
                  hoveredArtifact === artifact.id
                    ? "opacity-100"
                    : "opacity-0 group-hover:opacity-100"
                }`}
              >
                <Download className="w-4 h-4" />
              </button>
            </div>
            ))
          )}
        </div>

        {/* Content Section */}
        {contentItems.length > 0 && (
          <div className="px-4 pb-4">
            <div className="flex items-center justify-between mb-3 mt-2">
              <h4 className="text-[14px] text-[#e0e0e0]">Content</h4>
              <ChevronRight className="w-4 h-4 text-[#555]" />
            </div>
            <div className="grid grid-cols-2 gap-2">
              {contentItems.map((item) => (
                <div
                  key={item.id}
                  onClick={() => onContentClick?.(item.id)}
                  className="relative group rounded-lg overflow-hidden bg-[#111114] border border-[#2a2a2e] hover:border-[#3a3a42] transition-all cursor-pointer aspect-[16/10]"
                >
                  {/* Fake terminal content */}
                  <div className="absolute inset-0 p-2 flex flex-col gap-1">
                    <div className="h-1 w-[60%] bg-[#2a3a2a] rounded-full" />
                    <div className="h-1 w-[80%] bg-[#2a2a3a] rounded-full" />
                    <div className="h-1 w-[45%] bg-[#3a2a2a] rounded-full" />
                    <div className="h-1 w-[70%] bg-[#2a3a3a] rounded-full" />
                    <div className="h-1 w-[55%] bg-[#2a2a3a] rounded-full" />
                    <div className="h-1 w-[65%] bg-[#2a3a2a] rounded-full" />
                  </div>

                  {/* Video play button */}
                  {item.isVideo && (
                    <div className="absolute bottom-1.5 right-1.5 w-6 h-6 rounded-full bg-[#6366f1]/80 flex items-center justify-center">
                      <Play className="w-3 h-3 text-white fill-white" />
                    </div>
                  )}

                  {/* Hover overlay */}
                  <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Bottom scrollbar indicator */}
      <div className="px-4 pb-3 pt-1">
        <div className="w-full h-1 rounded-full bg-[#222226] overflow-hidden">
          <div className="h-full w-[30%] rounded-full bg-[#3a3a42]" />
        </div>
      </div>
    </div>
  );
}
