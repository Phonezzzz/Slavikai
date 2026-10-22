import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { ModelQualification } from '../components/model-qualification';

afterEach(() => { vi.unstubAllGlobals(); });

it('sends an explicit inert setup request and shows only server evidence', async () => {
  const fetcher = vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ state: 'unknown', can_check: true }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ state: 'supported', can_check: true }) });
  vi.stubGlobal('fetch', fetcher);
  const sessionId = 'session-a';
  render(<ModelQualification sessionId={sessionId} sessionHeader="X-Session" />);
  fireEvent.click(await screen.findByRole('button', { name: 'Проверить tools' }));
  await screen.findByText('Tools: проверены для этой сессии');
  expect(fetcher.mock.calls[1][0]).toBe(`/ui/api/sessions/${sessionId}/model/qualify`);
  expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ confirm: true });
});

it('hides an unauthorized or unknown action', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false }));
  const { container } = render(<ModelQualification sessionId="member" sessionHeader="X-Session" />);
  await act(async () => {});
  expect(container.innerHTML).toBe('');
});

it('aborts a pending check on unmount and ignores its late result', async () => {
  let resolve: (value: unknown) => void = () => {};
  const fetcher = vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ state: 'unknown', can_check: true }) })
    .mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
  vi.stubGlobal('fetch', fetcher);
  const { unmount } = render(<ModelQualification sessionId="a" sessionHeader="X-Session" />);
  fireEvent.click(await screen.findByRole('button', { name: 'Проверить tools' }));
  await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
  const signal = fetcher.mock.calls[1][1].signal;
  unmount();
  expect(signal.aborted).toBe(true);
  await act(async () => { resolve({ ok: true, json: async () => ({ state: 'supported', can_check: true }) }); });
});

it('refreshes read-only evidence after the model becomes idle', async () => {
  const fetcher = vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ state: 'unknown', can_check: false }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ state: 'unknown', can_check: true }) });
  vi.stubGlobal('fetch', fetcher);
  render(<ModelQualification sessionId="idle" sessionHeader="X-Session" />);
  fireEvent.click(await screen.findByRole('button', { name: 'Обновить доступность проверки' }));
  await screen.findByRole('button', { name: 'Проверить tools' });
  expect(fetcher).toHaveBeenCalledTimes(2);
  expect(fetcher.mock.calls.every((call) => !call[1].method)).toBe(true);
});
