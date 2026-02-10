import { useEffect, useState } from "react";
import { ArtifactsSidebar, type Artifact } from "./artifacts-sidebar";
import { DocumentViewer } from "./document-viewer";

type PanelView = "sidebar" | "viewer";

interface ArtifactPanelProps {
  isOpen: boolean;
  onClose: () => void;
  artifacts: Artifact[];
  autoOpenArtifactId?: string | null;
  onDownloadArtifact?: (artifact: Artifact) => void;
  onDownloadAll?: () => void;
  className?: string;
}

export function ArtifactPanel({
  isOpen,
  onClose,
  artifacts,
  autoOpenArtifactId = null,
  onDownloadArtifact,
  onDownloadAll,
  className = "",
}: ArtifactPanelProps) {
  const [view, setView] = useState<PanelView>("sidebar");
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);

  useEffect(() => {
    if (!autoOpenArtifactId) {
      return;
    }
    const target = artifacts.find((artifact) => artifact.id === autoOpenArtifactId);
    if (!target) {
      return;
    }
    setSelectedArtifact(target);
    setView("viewer");
  }, [autoOpenArtifactId, artifacts]);

  useEffect(() => {
    if (!selectedArtifact) {
      return;
    }
    const stillExists = artifacts.some((artifact) => artifact.id === selectedArtifact.id);
    if (!stillExists) {
      setSelectedArtifact(null);
      setView("sidebar");
    }
  }, [artifacts, selectedArtifact]);

  if (!isOpen) return null;

  const handleArtifactClick = (artifact: Artifact) => {
    setSelectedArtifact(artifact);
    setView("viewer");
  };

  const handleBack = () => {
    setView("sidebar");
    setSelectedArtifact(null);
  };

  const handleClose = () => {
    setView("sidebar");
    setSelectedArtifact(null);
    onClose();
  };

  if (view === "viewer" && selectedArtifact) {
    return (
      <div
        className={`h-full w-[40vw] min-w-[360px] max-w-[720px] flex-shrink-0 max-md:w-[88vw] max-md:min-w-0 ${className}`}
      >
        <DocumentViewer
          title={selectedArtifact.name}
          type={selectedArtifact.type}
          content={selectedArtifact.content || ""}
          artifactKind={selectedArtifact.artifactKind}
          fileName={selectedArtifact.fileName}
          language={selectedArtifact.language}
          onDownload={() => onDownloadArtifact?.(selectedArtifact)}
          onBack={handleBack}
          onClose={handleClose}
        />
      </div>
    );
  }

  return (
    <div
      className={`h-full w-[20vw] min-w-[280px] max-w-[420px] flex-shrink-0 max-md:w-[88vw] max-md:min-w-0 ${className}`}
    >
      <ArtifactsSidebar
        artifacts={artifacts}
        onArtifactClick={handleArtifactClick}
        onDownloadArtifact={(artifactId) => {
          const target = artifacts.find((artifact) => artifact.id === artifactId);
          if (!target) {
            return;
          }
          onDownloadArtifact?.(target);
        }}
        onDownloadAll={onDownloadAll}
        onClose={handleClose}
      />
    </div>
  );
}
