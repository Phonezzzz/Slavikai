import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { Settings } from '../components/Settings';
const config = { state: 'available', basis: 'loaded_default_model_at_refresh', source: 'Saved default model configuration with loader defaults', identity: { provider: 'local', model: 'saved-fixture' }, values: { temperature: 0.23, top_p: 0.61, max_tokens: 2345, mode: 'default', thinking_enabled: true, reasoning_effort: 'high', reasoning_summary: true, reasoning_summary_wait: false, diffusing: null }, presence: { api_key: true, base_url: true, extra_headers: true, system_prompt: true }, unavailable_fields: [], unsupported_fields: ['web_search_enabled'] };
const setup = (owner = true, value: unknown = config) => {
  const mock = vi.fn(async (url: string, _init?: RequestInit) => {
    if (url === '/ui/api/settings') return new Response(JSON.stringify({ settings: { can_manage_installation: owner, preferences_subject: owner ? 'owner' : 'member', preferences_etag: 'tag', model: { provider: 'local', model: 'saved-fixture' }, model_config_snapshot: value } }));
    throw new Error(url);
  });
  vi.stubGlobal('fetch', mock);
  return mock;
};
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const advanced = async () => {
  fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Advanced saved model options' }));
};
it('shows saved safe values and presence flags without dirtying settings or probing a model', async () => {
  const mock = setup();
  render(<Settings isOpen onClose={() => undefined} />);
  await advanced();
  const region = await screen.findByRole('region', { name: 'Saved model options' });
  expect(within(region).getByText('0.23')).toBeTruthy();
  expect(within(region).getByText('2345')).toBeTruthy();
  expect(within(region).getByText('high')).toBeTruthy();
  expect(within(region).getByText(/local.*saved-fixture/)).toBeTruthy();
  expect(within(region).getByText(/Already running chats may use different options/)).toBeTruthy();
  expect(within(region).getByText(/loader does not restore this field/)).toBeTruthy();
  expect(region.querySelector('input,textarea,select')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText('No changes to save.');
  expect(mock.mock.calls.some(([url, init]) => init?.method === 'POST' || url.startsWith('/ui/api/models') || url.includes('/capabilities'))).toBe(false);
});
it.each([null, { state: 'unavailable' }, { ...config, basis: 'active_chat' }])('does not invent saved options when unavailable', async value => {
  setup(true, value);
  render(<Settings isOpen onClose={() => undefined} />);
  await advanced();
  await screen.findByText('Saved model options are unavailable.');
  expect(screen.queryByText('0.7')).toBeNull();
});
it('never renders owner diagnostics to a member', async () => {
  setup(false);
  render(<Settings isOpen onClose={() => undefined} />);
  await screen.findByRole('button', { name: 'Memory and data' });
  expect(screen.queryByRole('button', { name: 'Models and connections' })).toBeNull();
  expect(screen.queryByText('0.23')).toBeNull();
});
it('keeps saved identity separate from an unsaved provider selection', async () => {
  const mock = setup();
  render(<Settings isOpen onClose={() => undefined} />);
  await advanced();
  fireEvent.change(screen.getByRole('combobox', { name: 'Provider' }), { target: { value: 'openrouter' } });
  const view = screen.getByRole('region', { name: 'Saved model options' });
  expect(within(view).getByText(/local.*saved-fixture/)).toBeTruthy();
  expect(mock.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false);
});
it('does not send the read-only snapshot when saving another Memory setting', async () => {
  const mock = setup();
  render(<Settings isOpen onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Memory and data' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Advanced memory settings' }));
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Inbox size' }), { target: { value: '321' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText('Settings saved.');
  const writes = mock.mock.calls.filter(([, init]) => init?.method === 'POST');
  expect(writes).toHaveLength(1);
  expect(JSON.parse(writes[0][1]?.body as string)).toEqual(expect.objectContaining({ memory: expect.objectContaining({ inbox_max_items: 321 }) }));
  expect(JSON.parse(writes[0][1]?.body as string).model_config_snapshot).toBeUndefined();
  expect(JSON.parse(writes[0][1]?.body as string).model).toBeUndefined();
});
it('does not echo untrusted non-scalar values from malformed diagnostics', async () => {
  setup(true, { ...config, values: { ...config.values, temperature: 'SECRET_VALUE', mode: { secret: 'SECRET_OBJECT' } } });
  const h = render(<Settings isOpen onClose={() => undefined} />);
  await advanced();
  expect(h.container.textContent).not.toContain('SECRET_');
  expect(screen.getAllByText('Unavailable').length).toBeGreaterThan(0);
});
