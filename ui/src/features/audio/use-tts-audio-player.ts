import { useCallback, useEffect, useRef, useState } from 'react';

export type TtsPlaybackStatus = 'idle' | 'loading' | 'playing' | 'paused' | 'error';

export type TtsPlaybackState = {
  activeMessageId: string | null;
  status: TtsPlaybackStatus;
  currentTime: number;
  duration: number;
  error: string | null;
};

const initialState: TtsPlaybackState = {
  activeMessageId: null,
  status: 'idle',
  currentTime: 0,
  duration: 0,
  error: null,
};

const extractErrorMessage = (payload: unknown, fallback: string): string => {
  if (!payload || typeof payload !== 'object') {
    return fallback;
  }
  const body = payload as { error?: { message?: unknown } };
  if (body.error && typeof body.error.message === 'string' && body.error.message.trim()) {
    return body.error.message;
  }
  return fallback;
};

const finiteOrZero = (value: number): number => (Number.isFinite(value) && value > 0 ? value : 0);

export function useTtsAudioPlayer(sessionId?: string | null) {
  const [state, setState] = useState<TtsPlaybackState>(initialState);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const requestSeqRef = useRef(0);

  const releaseAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
      audioRef.current = null;
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    requestSeqRef.current += 1;
    releaseAudio();
    setState(initialState);
  }, [releaseAudio]);

  useEffect(
    () => () => {
      requestSeqRef.current += 1;
      releaseAudio();
    },
    [releaseAudio],
  );

  const pause = useCallback(() => {
    if (!audioRef.current) {
      setState((prev) => (prev.status === 'playing' ? { ...prev, status: 'paused' } : prev));
      return;
    }
    audioRef.current.pause();
    setState((prev) => ({ ...prev, status: 'paused' }));
  }, []);

  const resume = useCallback(async () => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    try {
      await audio.play();
      setState((prev) => ({ ...prev, status: 'playing', error: null }));
    } catch (error) {
      releaseAudio();
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Не удалось воспроизвести TTS-аудио.',
      }));
    }
  }, [releaseAudio]);

  const seek = useCallback((nextTime: number) => {
    const normalizedTime = Math.max(0, finiteOrZero(nextTime));
    const audio = audioRef.current;
    if (audio) {
      const maxTime = finiteOrZero(audio.duration);
      audio.currentTime = maxTime > 0 ? Math.min(normalizedTime, maxTime) : normalizedTime;
    }
    setState((prev) => {
      const maxTime = finiteOrZero(prev.duration);
      return {
        ...prev,
        currentTime: maxTime > 0 ? Math.min(normalizedTime, maxTime) : normalizedTime,
      };
    });
  }, []);

  const toggle = useCallback(
    async (messageId: string, text: string) => {
      const trimmedText = text.trim();
      if (!trimmedText) {
        return;
      }

      if (state.activeMessageId === messageId && state.status === 'playing') {
        pause();
        return;
      }
      if (state.activeMessageId === messageId && state.status === 'paused') {
        await resume();
        return;
      }
      if (state.activeMessageId === messageId && state.status === 'loading') {
        return;
      }

      requestSeqRef.current += 1;
      const requestSeq = requestSeqRef.current;
      releaseAudio();
      setState({
        activeMessageId: messageId,
        status: 'loading',
        currentTime: 0,
        duration: 0,
        error: null,
      });

      try {
        const response = await fetch('/ui/api/tts/speak', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(sessionId ? { 'X-Slavik-Session': sessionId } : {}),
          },
          body: JSON.stringify({ text: trimmedText }),
        });
        if (!response.ok) {
          let payload: unknown = null;
          try {
            payload = await response.json();
          } catch {
            payload = null;
          }
          const fallback = `TTS request failed (HTTP ${response.status}).`;
          throw new Error(extractErrorMessage(payload, fallback));
        }
        const audioBlob = await response.blob();
        if (audioBlob.size === 0) {
          throw new Error('TTS returned empty audio.');
        }
        if (requestSeqRef.current !== requestSeq) {
          return;
        }

        const objectUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(objectUrl);
        audioRef.current = audio;
        audioUrlRef.current = objectUrl;

        audio.onloadedmetadata = () => {
          setState((prev) =>
            prev.activeMessageId === messageId
              ? { ...prev, duration: finiteOrZero(audio.duration) }
              : prev,
          );
        };
        audio.ontimeupdate = () => {
          setState((prev) =>
            prev.activeMessageId === messageId
              ? {
                  ...prev,
                  currentTime: finiteOrZero(audio.currentTime),
                  duration: finiteOrZero(audio.duration) || prev.duration,
                }
              : prev,
          );
        };
        audio.onended = () => {
          if (requestSeqRef.current === requestSeq) {
            stop();
          }
        };
        audio.onerror = () => {
          if (requestSeqRef.current === requestSeq) {
            releaseAudio();
            setState({
              activeMessageId: messageId,
              status: 'error',
              currentTime: 0,
              duration: 0,
              error: 'Не удалось воспроизвести аудио от TTS сервиса.',
            });
          }
        };

        await audio.play();
        if (requestSeqRef.current === requestSeq) {
          setState((prev) => ({
            ...prev,
            activeMessageId: messageId,
            status: 'playing',
            currentTime: finiteOrZero(audio.currentTime),
            duration: finiteOrZero(audio.duration),
            error: null,
          }));
        }
      } catch (error) {
        if (requestSeqRef.current !== requestSeq) {
          return;
        }
        releaseAudio();
        setState({
          activeMessageId: messageId,
          status: 'error',
          currentTime: 0,
          duration: 0,
          error: error instanceof Error ? error.message : 'TTS failed.',
        });
      }
    },
    [pause, releaseAudio, resume, sessionId, state.activeMessageId, state.status, stop],
  );

  return {
    state,
    toggle,
    pause,
    resume,
    seek,
    stop,
  };
}
