import { SESSION_MODE_LABELS, type ModeTransitionsContract, type SessionMode, type UiDecision } from '../types';

type ComposerModesProps = {
  sessionId: string | null;
  mode: SessionMode;
  transitions: ModeTransitionsContract | null;
  busy: boolean;
  decision?: UiDecision | null;
  onChangeMode: (mode: SessionMode) => Promise<void>;
};

export function ComposerModes({ sessionId, mode, transitions, busy, decision = null, onChangeMode }: ComposerModesProps) {
  const decisionBlocking = decision?.blocking === true
    && (decision.status === 'pending' || decision.status === 'executing');
  const disabled = !sessionId || busy || decisionBlocking || !transitions;
  const environment = mode === 'desktop' ? 'computer' : 'chat';
  const reason = (target: SessionMode) => transitions?.targets[target]?.message
    ?? transitions?.targets[target]?.reasonCode ?? undefined;
  return <div className="mb-2 flex flex-wrap items-center gap-2" aria-label="Способ работы">
    <div className="flex items-center gap-1" role="group" aria-label="Режим чата">
      {(['ask', 'auto', 'plan'] as const).map((target) => <button
        key={target} type="button" aria-pressed={mode === target}
        title={environment === 'computer' ? 'Для режимов чата выберите среду «Чат».' : reason(target)}
        disabled={disabled || environment === 'computer' || !transitions?.targets[target]?.allowed}
        onClick={() => { void onChangeMode(target); }}
        className={`rounded-lg border px-2 py-1.5 text-xs ${mode === target
          ? 'border-zinc-600 bg-zinc-800 text-zinc-100' : 'border-zinc-800 bg-zinc-900 text-zinc-400'} disabled:opacity-50`}
      >{SESSION_MODE_LABELS[target]}</button>)}
    </div>
    <label className="flex items-center gap-2 text-xs text-zinc-400">
      Среда
      <select aria-label="Рабочая среда" value={environment} disabled={disabled}
        onChange={(event) => { void onChangeMode(event.target.value === 'computer' ? 'desktop' : 'ask'); }}
        className="max-w-full rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-zinc-200">
        <option value="chat" disabled={environment !== 'chat' && !transitions?.targets.ask.allowed}>Чат</option>
        <option value="computer" disabled={environment !== 'computer' && !transitions?.targets.desktop.allowed}>Этот компьютер</option>
      </select>
    </label>
    {mode === 'act' ? <span className="text-xs text-zinc-400">Выполнение плана</span> : null}
    {environment === 'computer' ? <p className="w-full text-xs text-amber-200">
      Действия на этом компьютере доступны по его правилам доступа и подтверждениям.
    </p> : null}
    {environment === 'computer' && !transitions?.targets.ask.allowed && reason('ask')
      ? <p className="w-full text-xs text-zinc-400">{reason('ask')}</p> : null}
  </div>;
}
