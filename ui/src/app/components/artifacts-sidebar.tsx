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
  type: "MD" | "PY" | "JS" | "TS" | "JSON" | "TXT" | "HTML" | "CSS" | "SH";
  category: "Document" | "Code" | "Config" | "Script";
  content?: string;
  artifactKind?: "text" | "file";
  sourceArtifactId?: string | null;
  fileName?: string | null;
  fileExt?: string | null;
  language?: string | null;
  fileContent?: string | null;
  sessionFilePath?: string | null;
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

const defaultArtifacts: Artifact[] = [];
const defaultContentItems: ContentItem[] = [];

const typeColorMap: Record<string, string> = {
  MD: "text-blue-400",
  PY: "text-yellow-400",
  JS: "text-yellow-300",
  TS: "text-blue-300",
  JSON: "text-green-400",
  TXT: "text-gray-400",
  HTML: "text-orange-400",
  CSS: "text-purple-400",
  SH: "text-emerald-400",
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
  const previewItems = contentItems.filter((item) => Boolean(item.thumbnail));

  return (
    <div
      className={`flex h-full w-full flex-col bg-zinc-950 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4">
        <h3 className="text-zinc-200 text-[15px]">Artifacts</h3>
        <div className="flex items-center gap-3">
          <button
            onClick={onDownloadAll}
            className="flex items-center gap-1.5 text-[13px] text-zinc-400 hover:text-[#ccc] transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            Download all
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="text-zinc-500 hover:text-[#ccc] transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto" data-scrollbar="auto">
        <div className="p-4 space-y-2">
          {/* Artifacts List */}
          {artifacts.length === 0 ? (
            <div className="rounded-xl bg-zinc-900 p-4 text-[13px] text-zinc-500">
              Нет артефактов.
            </div>
          ) : (
            artifacts.map((artifact) => (
              <div
                key={artifact.id}
                className="group flex items-center gap-3 p-3 rounded-xl bg-zinc-900 hover:bg-zinc-800 transition-all cursor-pointer"
                onMouseEnter={() => setHoveredArtifact(artifact.id)}
                onMouseLeave={() => setHoveredArtifact(null)}
                onClick={() => onArtifactClick?.(artifact)}
              >
                <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-zinc-800">
                  {categoryIconMap[artifact.category]}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[14px] text-zinc-200 truncate">
                    {artifact.name}
                  </div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="text-[12px] text-zinc-500">
                      {artifact.category}
                    </span>
                    <span className="text-[12px] text-zinc-500">-</span>
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
                  className={`text-zinc-500 hover:text-[#ccc] transition-all ${
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
        {previewItems.length > 0 && (
          <div className="px-4 pb-4">
            <div className="flex items-center justify-between mb-3 mt-2">
              <h4 className="text-[14px] text-zinc-200">Content</h4>
              <ChevronRight className="w-4 h-4 text-zinc-500" />
            </div>
            <div className="grid grid-cols-2 gap-2">
              {previewItems.map((item) => (
                <div
                  key={item.id}
                  onClick={() => onContentClick?.(item.id)}
                  className="relative group rounded-lg overflow-hidden bg-zinc-900 transition-all cursor-pointer aspect-[16/10]"
                >
                  {item.thumbnail ? (
                    <img
                      src={item.thumbnail}
                      alt={item.title ?? "Preview"}
                      className="absolute inset-0 h-full w-full object-cover"
                    />
                  ) : null}

                  {/* Video play button */}
                  {item.isVideo && (
                    <div className="absolute bottom-1.5 right-1.5 w-6 h-6 rounded-full bg-black/50 flex items-center justify-center">
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
    </div>
  );
}
