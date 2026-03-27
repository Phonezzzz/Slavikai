import { useEffect, useState, type Dispatch, type SetStateAction } from 'react';

import type { AppView } from './session-storage';

type UseSessionOverlaysOptions = {
  activeView: AppView;
};

export type SessionOverlaysResult = {
  artifactPanelOpen: boolean;
  artifactViewerArtifactId: string | null;
  searchOpen: boolean;
  settingsOpen: boolean;
  sessionDrawerOpen: boolean;
  repositoryPanelOpen: boolean;
  forceCanvasNext: boolean;
  setArtifactPanelOpen: (value: boolean) => void;
  setArtifactViewerArtifactId: (value: string | null) => void;
  setSearchOpen: (value: boolean) => void;
  setSettingsOpen: (value: boolean) => void;
  setSessionDrawerOpen: (value: boolean) => void;
  setRepositoryPanelOpen: (value: boolean) => void;
  setForceCanvasNext: Dispatch<SetStateAction<boolean>>;
  openStreamedArtifact: (artifactId: string) => void;
  resetSessionSurfaceState: () => void;
};

export function useSessionOverlays({
  activeView,
}: UseSessionOverlaysOptions): SessionOverlaysResult {
  const [artifactPanelOpen, setArtifactPanelOpen] = useState(false);
  const [artifactViewerArtifactId, setArtifactViewerArtifactId] = useState<string | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sessionDrawerOpen, setSessionDrawerOpen] = useState(false);
  const [repositoryPanelOpen, setRepositoryPanelOpen] = useState(false);
  const [forceCanvasNext, setForceCanvasNext] = useState(false);

  useEffect(() => {
    if (activeView === 'workspace') {
      setArtifactPanelOpen(false);
    } else {
      setRepositoryPanelOpen(false);
    }
  }, [activeView]);

  const openStreamedArtifact = (artifactId: string) => {
    setArtifactPanelOpen(true);
    setArtifactViewerArtifactId(artifactId);
  };

  const resetSessionSurfaceState = () => {
    setArtifactViewerArtifactId(null);
  };

  return {
    artifactPanelOpen,
    artifactViewerArtifactId,
    searchOpen,
    settingsOpen,
    sessionDrawerOpen,
    repositoryPanelOpen,
    forceCanvasNext,
    setArtifactPanelOpen,
    setArtifactViewerArtifactId,
    setSearchOpen,
    setSettingsOpen,
    setSessionDrawerOpen,
    setRepositoryPanelOpen,
    setForceCanvasNext,
    openStreamedArtifact,
    resetSessionSurfaceState,
  };
}
