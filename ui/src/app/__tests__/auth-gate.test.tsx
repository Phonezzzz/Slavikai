import { cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import AuthGate from '../AuthGate';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('AuthGate', () => {
  it('renders the application after cookie auth is confirmed', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ authenticated: true, auth_required: true }), {
          status: 200,
        }),
      ),
    );

    const view = render(
      <AuthGate>
        <div>application-ready</div>
      </AuthGate>,
    );

    await vi.waitFor(() => expect(view.container.textContent).toContain('application-ready'));
  });

  it('logs in with the token and never stores it in frontend state after success', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ authenticated: false, auth_required: true }), {
          status: 200,
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ authenticated: true }), { status: 200 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const view = render(
      <AuthGate>
        <div>application-ready</div>
      </AuthGate>,
    );
    const input = await vi.waitFor(() => view.getByLabelText('API token'));
    fireEvent.change(input, { target: { value: 'secret-token' } });
    const form = input.closest('form');
    if (!form) throw new Error('Auth form not found');
    fireEvent.submit(form);

    await vi.waitFor(() => expect(view.container.textContent).toContain('application-ready'));
    expect(fetchMock).toHaveBeenLastCalledWith(
      '/ui/api/auth/login',
      expect.objectContaining({
        credentials: 'same-origin',
        body: JSON.stringify({ token: 'secret-token' }),
      }),
    );
  });
});
