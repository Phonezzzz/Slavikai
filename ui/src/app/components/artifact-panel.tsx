import { useEffect, useState } from "react";
import { ArtifactsSidebar, type Artifact } from "./artifacts-sidebar";
import { DocumentViewer } from "./document-viewer";

type PanelView = "sidebar" | "viewer";

interface ArtifactPanelProps {
  isOpen: boolean;
  onClose: () => void;
  artifacts: Artifact[];
  autoOpenArtifactId?: string | null;
  className?: string;
}

export function ArtifactPanel({
  isOpen,
  onClose,
  artifacts,
  autoOpenArtifactId = null,
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
      <div className={`w-[50vw] max-w-[700px] min-w-[400px] flex-shrink-0 h-full ${className}`}>
        <DocumentViewer
          title={selectedArtifact.name}
          type={selectedArtifact.type}
          content={selectedArtifact.content || ""}
          onBack={handleBack}
          onClose={handleClose}
        />
      </div>
    );
  }

  return (
    <div className={`flex-shrink-0 h-full ${className}`}>
      <ArtifactsSidebar
        artifacts={artifacts}
        onArtifactClick={handleArtifactClick}
        onClose={handleClose}
      />
    </div>
  );
}
