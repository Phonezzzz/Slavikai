import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { RecoveryRunsView } from '../components/recovery-runs-view';

const json = (body: unknown) => new Response(JSON.stringify(body));
const session = 'A';
const run = { task_run_id: 'run-a', recovery_prior_state: 'running', recovery_occurred_at: '2026-11-07' };
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it('reads the selected run and pages its evidence using the originating session', async () => {
  const fetch = vi.fn(async (url: string, _init?: RequestInit) => {
    if (url.endsWith('/recovery')) return json({ runs: [run], next_cursor: 'run-cursor' });
    if (url.endsWith('?after=run-cursor')) {
      return json({ runs: [{ ...run, task_run_id: 'run-b' }], next_cursor: null });
    }
    if (url.endsWith('?after=cursor')) return json({ operations: [], next_cursor: null, external_effects: 'unreconciled' });
    return json({
      operations: [{
        operation_id: 'op', tool_attempt_id: 'attempt', request_path: 'script.py',
        publication: 'published', publication_source: 'observation',
        publication_snapshot_sha256: 'digest', publication_snapshot_source: 'workspace_root_marker',
        publication_cleanup_warning: true,
        observation: {
          exit_code: null, capture_error: null, timed_out: true,
          stdout_bytes_seen: 20, stdout_bytes_retained: 10, stdout_eof: true,
          stderr_bytes_seen: 3, stderr_bytes_retained: 3, stderr_eof: false,
        },
      }],
      next_cursor: 'cursor', external_effects: 'unreconciled',
    });
  });
  vi.stubGlobal('fetch', fetch);
  render(<RecoveryRunsView sessionId="A" sessionHeader="X-Test" />);
  fireEvent.click(screen.getByRole('button', { name: 'Показать запуски для сверки' }));
  fireEvent.click(await screen.findByRole('button', { name: /run-a/ }));
  expect(await screen.findByText(/SHA-256: digest/)).toBeTruthy();
  expect(screen.getByText('Время выполнения истекло.')).toBeTruthy();
  expect(screen.getByText('После публикации очистка завершилась с ошибкой.')).toBeTruthy();
  expect(screen.getByText('stdout: сохранено 10 из 20 прочитанных байтов · EOF достигнут')).toBeTruthy();
  expect(screen.getByText('stderr: сохранено 3 из 3 прочитанных байтов · EOF не достигнут')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Следующая страница свидетельств' }));
  expect(screen.queryByText('Сохранённых попыток нет.')).toBeNull();
  await act(async () => {});
  expect(screen.getByText(/SHA-256: digest/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Следующая страница запусков' }));
  await screen.findByRole('button', { name: /run-b/ });
  expect(screen.getByRole('button', { name: /run-a/ })).toBeTruthy();
  expect(screen.getByText(/SHA-256: digest/)).toBeTruthy();
  expect(fetch.mock.calls.map(([url]) => url)).toEqual([
    `/ui/api/sessions/${session}/runs/recovery`,
    `/ui/api/sessions/${session}/runs/${run.task_run_id}/recovery-evidence`,
    `/ui/api/sessions/${session}/runs/${run.task_run_id}/recovery-evidence?after=cursor`,
    `/ui/api/sessions/${session}/runs/recovery?after=run-cursor`,
  ]);
  for (const [, init] of fetch.mock.calls) {
    expect(init?.headers).toEqual({ 'X-Test': 'A' });
    expect(init?.method ?? 'GET').toBe('GET');
  }
});

it('aborts an old session read and discards a response that arrives after switching', async () => {
  let finish!: (response: Response) => void;
  const fetch = vi.fn((_url: string, _init?: RequestInit) => new Promise<Response>((resolve) => {
    finish = resolve;
  }));
  vi.stubGlobal('fetch', fetch);
  const { rerender } = render(<RecoveryRunsView key="A" sessionId="A" sessionHeader="X-Test" />);
  fireEvent.click(screen.getByRole('button', { name: 'Показать запуски для сверки' }));
  rerender(<RecoveryRunsView key="B" sessionId="B" sessionHeader="X-Test" />);
  expect(fetch.mock.calls[0][1]?.signal?.aborted).toBe(true);
  await act(async () => finish(json({ runs: [run], next_cursor: null })));
  expect(screen.queryByRole('button', { name: /run-a/ })).toBeNull();
});
