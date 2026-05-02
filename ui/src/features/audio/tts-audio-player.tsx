import { LoaderCircle, Pause, Play, Square } from 'lucide-react';

import type { TtsPlaybackState } from './use-tts-audio-player';

type TtsAudioPlayerProps = {
  playback: TtsPlaybackState;
  onPlayPause: () => void;
  onSeek: (time: number) => void;
  onStop: () => void;
};

const formatTime = (value: number): string => {
  const seconds = Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
  const minutesPart = Math.floor(seconds / 60);
  const secondsPart = seconds % 60;
  return `${minutesPart}:${secondsPart.toString().padStart(2, '0')}`;
};

export function TtsAudioPlayer({ playback, onPlayPause, onSeek, onStop }: TtsAudioPlayerProps) {
  const isLoading = playback.status === 'loading';
  const isPlaying = playback.status === 'playing';
  const canSeek = playback.duration > 0 && playback.status !== 'loading' && playback.status !== 'error';
  const progressMax = Math.max(playback.duration, playback.currentTime, 1);

  return (
    <div className="mt-2 flex w-full max-w-[520px] items-center gap-2 rounded-md border border-white/[0.06] bg-white/[0.025] px-2.5 py-2 text-[11px] text-zinc-500">
      <button
        type="button"
        onClick={onPlayPause}
        disabled={isLoading || playback.status === 'error'}
        className="inline-flex h-6 w-6 flex-shrink-0 items-center justify-center rounded text-zinc-400 transition-colors hover:bg-white/[0.06] hover:text-zinc-100 disabled:cursor-wait disabled:opacity-70"
        title={isPlaying ? 'Pause' : 'Play'}
        aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
      >
        {isLoading ? (
          <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
        ) : isPlaying ? (
          <Pause className="h-3.5 w-3.5" />
        ) : (
          <Play className="h-3.5 w-3.5" />
        )}
      </button>

      <span className="w-9 flex-shrink-0 text-right font-mono tabular-nums">
        {formatTime(playback.currentTime)}
      </span>

      <input
        type="range"
        min={0}
        max={progressMax}
        step={0.1}
        value={Math.min(playback.currentTime, progressMax)}
        disabled={!canSeek}
        onChange={(event) => onSeek(Number(event.currentTarget.value))}
        className="h-1 min-w-0 flex-1 cursor-pointer accent-zinc-300 disabled:cursor-not-allowed disabled:opacity-40"
        aria-label="Audio position"
      />

      <span className="w-9 flex-shrink-0 font-mono tabular-nums">
        {formatTime(playback.duration)}
      </span>

      <button
        type="button"
        onClick={onStop}
        className="inline-flex h-6 w-6 flex-shrink-0 items-center justify-center rounded text-zinc-500 transition-colors hover:bg-white/[0.06] hover:text-zinc-100"
        title="Stop"
        aria-label="Stop audio"
      >
        <Square className="h-3.5 w-3.5" />
      </button>

      {playback.status === 'error' && playback.error ? (
        <span className="min-w-0 flex-1 truncate text-rose-300" title={playback.error}>
          {playback.error}
        </span>
      ) : null}
    </div>
  );
}
