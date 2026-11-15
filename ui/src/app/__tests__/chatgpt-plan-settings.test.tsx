import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

import { Settings } from '../components/Settings';

const accountId = '0123456789abcdef0123456789abcdef';
const authorizationUrl = 'https://auth.openai.com/api/accounts/authorize?state=fixture';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function fakeSettings({ owner = true, available = true, loginUrl = authorizationUrl } = {}) {
  let accounts: object[] = [];
  let authorizationStatus: string | null = null;
  const reply = (payload: unknown) => Response.json(payload);
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url === '/ui/api/settings') {
      return reply({ settings: {
        can_manage_installation: owner,
        chatgpt_plan_available: available,
        preferences_etag: '"fixture"',
      } });
    }
    if (url === '/ui/api/embeddings/status') return reply({ state: 'ready' });
    if (url === '/ui/api/models?skip_custom_probe=1&discovery_only=1') return reply({ providers: [] });
    if (url === '/ui/api/chatgpt-plan') return reply({ status: 'enabled', accounts, authorization_status: authorizationStatus });
    if (url === '/ui/api/chatgpt-plan/authorize' && init?.method === 'POST') {
      return reply({ authorization_url: loginUrl });
    }
    if (url === '/ui/api/chatgpt-plan/disconnect' && init?.method === 'POST') {
      accounts = [{ id: accountId, label: 'Рабочий аккаунт', provider: null, status: 'disconnected' }];
      return reply({ revoked: false });
    }
    throw new Error(`Unexpected request: ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  const onRefreshModels = vi.fn(async () => {});
  render(<Settings isOpen onClose={() => {}} onRefreshModels={onRefreshModels} />);
  return {
    fetchMock,
    onRefreshModels,
    completeAuthorization: () => {
      authorizationStatus = 'connected';
      accounts = [{
        id: accountId,
        label: 'Рабочий аккаунт',
        provider: `chatgpt-plan-${accountId}`,
        status: 'connected',
      }];
    },
  };
}

it('connects and manages an account explicitly through Settings', async () => {
  const { fetchMock, onRefreshModels, completeAuthorization } = fakeSettings();
  const popup = { opener: window, closed: false, close: vi.fn(), location: { replace: vi.fn() } };
  vi.spyOn(window, 'open').mockImplementation(() => popup as unknown as Window);

  fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Войти через ChatGPT' }));
  await vi.waitFor(() => expect(popup.location.replace).toHaveBeenCalledWith(authorizationUrl));
  expect(popup.opener).toBeNull();
  expect(screen.queryByText('Рабочий аккаунт')).toBeNull();
  completeAuthorization();
  fireEvent.click(screen.getByRole('button', { name: 'Обновить аккаунты и модели' }));

  const account = await screen.findByRole('group', { name: 'Рабочий аккаунт' });
  expect(screen.getByText('Последний вход через ChatGPT подтверждён.')).toBeTruthy();
  await vi.waitFor(() => expect(onRefreshModels).toHaveBeenCalledOnce());
  fireEvent.click(within(account).getByRole('button', { name: 'Войти повторно' }));
  await vi.waitFor(() => expect(popup.location.replace).toHaveBeenCalledTimes(2));
  fireEvent.click(within(account).getByRole('button', { name: 'Отключить' }));
  await screen.findByText('Доступ в SlavikAI отключён. OpenAI не подтвердил отзыв разрешения.');
  await within(account).findByText('Отключён');
  expect(within(account).queryByRole('button', { name: 'Отключить' })).toBeNull();
  expect(onRefreshModels).toHaveBeenCalledTimes(2);
  const authBodies = fetchMock.mock.calls
    .filter(([url]) => url === '/ui/api/chatgpt-plan/authorize')
    .map(([, init]) => JSON.parse(String(init?.body)));
  expect(authBodies).toEqual([{}, { account_id: accountId, reconsent: true }]);
  const disconnect = fetchMock.mock.calls.find(([url]) => url === '/ui/api/chatgpt-plan/disconnect');
  expect(JSON.parse(String(disconnect?.[1]?.body))).toEqual({ account_id: accountId });
});

it('shows unavailable to the owner without starting account authorization', async () => {
  const { fetchMock } = fakeSettings({ available: false });
  fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
  await screen.findByText('Подписка ChatGPT недоступна на этой установке.');
  expect(screen.queryByRole('button', { name: 'Войти через ChatGPT' })).toBeNull();
  expect(fetchMock.mock.calls.some(([url]) => url.startsWith('/ui/api/chatgpt-plan'))).toBe(false);
});

it('does not display subscription controls for members', async () => {
  const { fetchMock } = fakeSettings({ owner: false });
  await screen.findByRole('textbox', { name: 'Custom instructions' });
  expect(screen.queryByRole('button', { name: 'Models and connections' })).toBeNull();
  expect(fetchMock.mock.calls.some(([url]) => url.startsWith('/ui/api/chatgpt-plan'))).toBe(false);
});

it('offers a safe link when the browser blocks the sign-in tab', async () => {
  fakeSettings();
  vi.spyOn(window, 'open').mockReturnValue(null);
  fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Войти через ChatGPT' }));
  const link = await screen.findByRole('link', { name: 'Открыть страницу входа OpenAI' });
  expect(link.getAttribute('href')).toBe(authorizationUrl);
  expect(link.getAttribute('target')).toBe('_blank');
  expect(link.getAttribute('rel')).toBe('noopener noreferrer');
});

it('rejects an authorization URL with a different origin', async () => {
  fakeSettings({ loginUrl: 'https://auth.openai.com.example.test/api/accounts/authorize?state=fixture' });
  const popup = { opener: window, closed: false, close: vi.fn(), location: { replace: vi.fn() } };
  vi.spyOn(window, 'open').mockImplementation(() => popup as unknown as Window);
  fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Войти через ChatGPT' }));
  await screen.findByText('Сервер не вернул допустимую ссылку входа OpenAI.');
  expect(popup.location.replace).not.toHaveBeenCalled();
  expect(popup.close).toHaveBeenCalledOnce();
});
