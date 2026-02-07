import { useState } from "react";
import { ArtifactsSidebar, type Artifact } from "./artifacts-sidebar";
import { DocumentViewer } from "./document-viewer";

type PanelView = "sidebar" | "viewer";

interface ArtifactPanelProps {
  isOpen: boolean;
  onClose: () => void;
  artifacts: Artifact[];
  className?: string;
}

export function ArtifactPanel({
  isOpen,
  onClose,
  artifacts,
  className = "",
}: ArtifactPanelProps) {
  const [view, setView] = useState<PanelView>("sidebar");
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);

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
