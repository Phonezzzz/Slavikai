import { type FormEvent, type ReactNode, useEffect, useState } from "react";

type AuthState = "loading" | "authenticated" | "required" | "error";

interface AuthStatusPayload {
  authenticated?: boolean;
  auth_required?: boolean;
}

export default function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>("loading");
  const [token, setToken] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    void fetch("/ui/api/auth/status", { credentials: "same-origin" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload: AuthStatusPayload = await response.json();
        setState(payload.authenticated || !payload.auth_required ? "authenticated" : "required");
      })
      .catch((error: unknown) => {
        setMessage(error instanceof Error ? error.message : "Auth status unavailable");
        setState("error");
      });
  }, []);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("");
    const response = await fetch("/ui/api/auth/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    if (!response.ok) {
      setMessage(response.status === 401 ? "Неверный API token" : `Ошибка входа: HTTP ${response.status}`);
      return;
    }
    setToken("");
    setState("authenticated");
  }

  if (state === "authenticated") return children;
  return (
    <main className="auth-gate">
      <section className="auth-card">
        <h1>SlavikAI</h1>
        {state === "loading" ? <p>Проверка доступа…</p> : null}
        {state === "error" ? <p className="auth-error">Сервер недоступен: {message}</p> : null}
        {state === "required" ? (
          <form onSubmit={(event) => void handleLogin(event)}>
            <label htmlFor="slavik-api-token">API token</label>
            <input
              id="slavik-api-token"
              type="password"
              autoComplete="current-password"
              value={token}
              onChange={(event) => setToken(event.target.value)}
              required
              autoFocus
            />
            {message ? <p className="auth-error">{message}</p> : null}
            <button type="submit">Войти</button>
          </form>
        ) : null}
      </section>
    </main>
  );
}
