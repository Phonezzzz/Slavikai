import { useEffect, useRef, useState } from 'react';
import { extractErrorMessage } from '../session-payload';

type Account = { id: string; label: string; provider: string | null; status: string };
type Snapshot = {
  status: 'enabled' | 'unavailable';
  accounts: Account[];
  authorization_status: string | null;
};

const API_PATH = '/ui/api/chatgpt-plan';
const BUTTON = 'rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-100 hover:bg-zinc-800 disabled:opacity-50';
const ACCOUNT_STATUS = new Map([
  ['connected', 'Подключён'],
  ['disabled', 'Доступ по подписке не разрешён'],
  ['expired', 'Требуется повторный вход'],
  ['disconnected', 'Отключён'],
]);
const AUTHORIZATION_STATUS = new Map([
  ['pending', 'Вход через ChatGPT ожидает завершения.'],
  ['connected', 'Последний вход через ChatGPT подтверждён.'],
  ['disabled', 'При последнем входе OpenAI не выдал разрешение на доступ по подписке.'],
  ['expired', 'Время для завершения входа через ChatGPT истекло. Начните вход заново.'],
  ['failed', 'Последний вход через ChatGPT не выполнен. Начните вход заново.'],
]);

async function request(path: string, signal: AbortSignal, body?: object): Promise<unknown> {
  const response = await fetch(`${API_PATH}${path}`, {
    signal,
    cache: 'no-store',
    ...(body && { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, 'Операция с аккаунтом ChatGPT не выполнена.'));
  }
  return payload;
}

function isAccount(value: unknown): value is Account {
  if (!value || typeof value !== 'object') return false;
  const account = value as Partial<Account>;
  return typeof account.id === 'string' && typeof account.label === 'string'
    && (account.provider === null || typeof account.provider === 'string')
    && typeof account.status === 'string';
}

function parseSnapshot(payload: unknown): Snapshot {
  if (payload && typeof payload === 'object') {
    const { status, accounts, authorization_status } = payload as Partial<Snapshot>;
    if ((status === 'enabled' || status === 'unavailable') && Array.isArray(accounts) && accounts.every(isAccount)) {
      return { status, accounts, authorization_status: typeof authorization_status === 'string' ? authorization_status : null };
    }
  }
  throw new Error('Сервер не подтвердил статус аккаунтов ChatGPT.');
}

function parseAuthorizationUrl(payload: unknown): string {
  const value = payload && typeof payload === 'object'
    ? (payload as { authorization_url?: unknown }).authorization_url
    : null;
  if (typeof value === 'string') {
    try {
      const url = new URL(value);
      if (url.origin === 'https://auth.openai.com' && url.pathname === '/api/accounts/authorize' && !url.username && !url.password) {
        return url.href;
      }
    } catch {
      // Reject malformed URLs with the same message as unexpected destinations.
    }
  }
  throw new Error('Сервер не вернул допустимую ссылку входа OpenAI.');
}

type Props = {
  available: boolean;
  onRefreshModels?: () => Promise<unknown>;
};

export function ChatGptPlanAccounts({ available, onRefreshModels }: Props) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [authorizationLink, setAuthorizationLink] = useState<string | null>(null);
  const operation = useRef<AbortController | null>(null);

  const loadAccounts = async (signal: AbortSignal) => {
    const next = parseSnapshot(await request('', signal));
    if (!signal.aborted) setSnapshot(next);
  };

  const run = async (action: (signal: AbortSignal) => Promise<void>) => {
    if (operation.current) return;
    const controller = new AbortController();
    operation.current = controller;
    setBusy(true);
    setError(null);
    setNotice(null);
    setAuthorizationLink(null);
    try {
      await action(controller.signal);
    } catch (error) {
      if (!controller.signal.aborted) {
        setError(error instanceof Error ? error.message : 'Операция с аккаунтом ChatGPT не выполнена.');
      }
    } finally {
      if (operation.current === controller) {
        operation.current = null;
        setBusy(false);
      }
    }
  };

  useEffect(() => {
    setSnapshot(null);
    if (available) void run(loadAccounts);
    return () => {
      operation.current?.abort();
      operation.current = null;
    };
  }, [available]);

  const refresh = async (signal: AbortSignal) => {
    const [accounts, models] = await Promise.allSettled([loadAccounts(signal), onRefreshModels?.()]);
    if (accounts.status === 'rejected') throw accounts.reason;
    if (models.status === 'rejected') throw new Error('Не удалось обновить каталог моделей. Повторите обновление.');
  };

  const authorize = (accountId?: string) => {
    void run(async (signal) => {
      const tab = window.open('', '_blank');
      if (tab) tab.opener = null;
      try {
        const body = accountId ? { account_id: accountId, reconsent: true } : {};
        const url = parseAuthorizationUrl(await request('/authorize', signal, body));
        if (signal.aborted) {
          tab?.close();
          return;
        }
        if (tab && !tab.closed) {
          tab.location.replace(url);
        } else {
          setAuthorizationLink(url);
        }
        setSnapshot((current) => current ? { ...current, authorization_status: 'pending' } : current);
        setNotice('Завершите вход на странице OpenAI. Когда откроется результат в SlavikAI, вернитесь сюда и нажмите «Обновить аккаунты и модели».');
      } catch (error) {
        tab?.close();
        throw error;
      }
    });
  };

  const disconnect = (accountId: string) => {
    void run(async (signal) => {
      const payload = await request('/disconnect', signal, { account_id: accountId });
      if (signal.aborted) return;
      const revoked = (payload as { revoked?: unknown } | null)?.revoked;
      setNotice(revoked === true
        ? 'Доступ в SlavikAI отключён. OpenAI подтвердил отзыв разрешения.'
        : 'Доступ в SlavikAI отключён. OpenAI не подтвердил отзыв разрешения.');
      await refresh(signal);
    });
  };

  if (!available) return <p className="text-xs text-zinc-400">Подписка ChatGPT недоступна на этой установке.</p>;

  const enabled = snapshot?.status === 'enabled';
  const authorizationStatus = AUTHORIZATION_STATUS.get(snapshot?.authorization_status ?? '');
  return (
    <div className="space-y-3 text-xs">
      <div className="flex flex-wrap gap-2">
        {enabled ? <button type="button" className={BUTTON} disabled={busy} onClick={() => authorize()}>
          {snapshot.accounts.length ? 'Добавить аккаунт ChatGPT' : 'Войти через ChatGPT'}
        </button> : null}
        <button type="button" className={BUTTON} disabled={busy} onClick={() => { void run(refresh); }}>
          Обновить аккаунты и модели
        </button>
      </div>
      {busy ? <p role="status" className="text-zinc-400">Выполняется операция с подпиской ChatGPT…</p> : null}
      {snapshot?.status === 'unavailable' ? <p className="text-zinc-400">Подписка ChatGPT недоступна на этой установке.</p> : null}
      {enabled && authorizationStatus ? <p role="status" className="text-zinc-400">{authorizationStatus}</p> : null}
      {notice ? <p role="status" className="text-zinc-300">{notice}</p> : null}
      {error ? <p role="alert" className="text-amber-200">{error}</p> : null}
      {authorizationLink ? <a href={authorizationLink} target="_blank" rel="noopener noreferrer" className="inline-block text-zinc-100 underline">
        Открыть страницу входа OpenAI
      </a> : null}
      {enabled ? snapshot.accounts.map((account) => (
        <div key={account.id} role="group" aria-label={account.label} className="flex flex-wrap items-center justify-between gap-2 border-t border-zinc-800 pt-3">
          <div>
            <p className="font-medium text-zinc-100">{account.label}</p>
            <p className="mt-1 text-zinc-400">{ACCOUNT_STATUS.get(account.status) ?? 'Статус неизвестен'}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" className={BUTTON} disabled={busy} onClick={() => authorize(account.id)}>Войти повторно</button>
            {account.provider !== null || account.status === 'disabled' ? <button type="button" className={BUTTON} disabled={busy} onClick={() => disconnect(account.id)}>
              Отключить
            </button> : null}
          </div>
        </div>
      )) : null}
    </div>
  );
}
