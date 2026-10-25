import { type FormEvent, type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { BrowserSubjectProvider, verifiedBrowserSubject } from './browser-subject';

type AuthState = "loading" | "authenticated" | "required" | "error";
interface AuthStatusPayload { authenticated?: boolean; auth_required?: boolean; browser_subject?: unknown }

export default function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>("loading");
  const [subject, setSubject] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [message, setMessage] = useState("");
  const [loggingIn, setLoggingIn] = useState(false);
  const loginBusy = useRef(false);
  const active = useRef(true);
  const hasAuthenticatedSnapshot = useRef(false);
  const generation = useRef(0);
  const request = useRef<AbortController | null>(null);

  const checkStatus = useCallback(async (background = false) => {
    const epoch = ++generation.current;
    request.current?.abort();
    const controller = new AbortController(); request.current = controller;
    const current = () => active.current && epoch === generation.current;
    try {
      const response = await fetch('/ui/api/auth/status', { credentials: 'same-origin', cache: 'no-store', signal: controller.signal });
      if (response.status === 401 || response.status === 403) {
        if (current()) { hasAuthenticatedSnapshot.current = false; setSubject(null); setState('error'); setMessage(`HTTP ${response.status}`); }
        return;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload: AuthStatusPayload = await response.json();
      if (!current()) return;
      const authenticated = payload.authenticated === true || payload.auth_required === false;
      hasAuthenticatedSnapshot.current = authenticated;
      setSubject(authenticated ? verifiedBrowserSubject(payload.browser_subject) : null);
      setState(authenticated ? 'authenticated' : 'required');
      setMessage('');
    } catch (error) {
      if (!current()) return;
      if (background && hasAuthenticatedSnapshot.current) { setMessage('Не удалось перепроверить доступ.'); return; }
      setSubject(null);
      setMessage(error instanceof Error ? error.message : 'Auth status unavailable');
      setState('error');
    }
  }, []);

  useEffect(() => {
    active.current = true;
    void checkStatus();
    const refresh = () => { if (!loginBusy.current && document.visibilityState !== 'hidden') void checkStatus(true); };
    window.addEventListener('focus', refresh);
    document.addEventListener('visibilitychange', refresh);
    return () => {
      active.current = false; generation.current += 1; request.current?.abort();
      window.removeEventListener('focus', refresh);
      document.removeEventListener('visibilitychange', refresh);
    };
  }, [checkStatus]);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (loginBusy.current) return;
    loginBusy.current = true; setLoggingIn(true); setMessage('');
    const epoch = ++generation.current; request.current?.abort();
    try {
      const response = await fetch('/ui/api/auth/login', {
        method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token }),
      });
      if (!active.current || epoch !== generation.current) return;
      if (!response.ok) { setMessage(response.status === 401 ? 'Неверный API token' : `Ошибка входа: HTTP ${response.status}`); return; }
      setToken('');
      await checkStatus();
    } catch (error) {
      if (active.current && epoch === generation.current) setMessage(error instanceof Error ? error.message : 'Ошибка входа');
    } finally {
      loginBusy.current = false;
      if (active.current) setLoggingIn(false);
    }
  }

  if (state === 'authenticated') return <>{message ? <div role="status" className="fixed left-1/2 top-3 z-[80] -translate-x-1/2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm text-zinc-200 shadow-lg">{message} <button type="button" onClick={() => void checkStatus(true)}>Повторить проверку</button></div> : null}<BrowserSubjectProvider subject={subject}>{children}</BrowserSubjectProvider></>;
  return <main className="auth-gate"><section className="auth-card">
    <h1>SlavikAI</h1>
    {state === 'loading' ? <p>Проверка доступа…</p> : null}
    {state === 'error' ? <><p className="auth-error">Сервер недоступен: {message}</p><button type="button" onClick={() => void checkStatus()}>Повторить</button></> : null}
    {state === 'required' ? <form onSubmit={event => void handleLogin(event)}>
      <label htmlFor="slavik-api-token">API token</label>
      <input id="slavik-api-token" type="password" autoComplete="current-password" value={token} onChange={event => setToken(event.target.value)} required autoFocus />
      {message ? <p className="auth-error">{message}</p> : null}
      <button type="submit" disabled={loggingIn}>{loggingIn ? 'Вход…' : 'Войти'}</button>
    </form> : null}
  </section></main>;
}
