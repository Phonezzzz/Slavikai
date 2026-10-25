import { afterEach, expect, it, vi } from 'vitest';
import { loadLastModel, loadLastSessionId, loadWorkspaceExplorerVisible, saveLastModel, saveLastSessionId, saveWorkspaceExplorerVisible } from '../session-storage';
const a = 'a'.repeat(64), b = 'b'.repeat(64);
afterEach(() => { localStorage.clear(); vi.restoreAllMocks(); });
it('isolates hints by verified subject and never reads or migrates unscoped hints', () => {
  localStorage.setItem('slavik.last.session', 'legacy-private');
  expect(loadLastSessionId(a)).toBeNull();
  saveLastSessionId(a, 'session-a');
  expect(loadLastSessionId(b)).toBeNull();
  saveLastSessionId(b, 'session-b');
  expect(loadLastSessionId(a)).toBe('session-a');
  expect(loadLastSessionId(b)).toBe('session-b');
  saveLastModel(a, { provider: 'local', model: 'private-a' });
  expect(loadLastModel(b)).toBeNull();
  expect(loadLastModel(a)).toEqual({ provider: 'local', model: 'private-a' });
  saveWorkspaceExplorerVisible(a, false);
  expect(loadWorkspaceExplorerVisible(a)).toBe(false);
  expect(loadWorkspaceExplorerVisible(b)).toBe(true);
  saveLastSessionId(a, null);
  expect(loadLastSessionId(a)).toBeNull();
  expect(loadLastSessionId(b)).toBe('session-b');
  expect(localStorage.getItem('slavik.last.session')).toBe('legacy-private');
});
it('disables persistence when the subject is missing or invalid', () => {
  const before = localStorage.length;
  for (const subject of [null, '', 'email:private@example.com']) {
    saveLastSessionId(subject, 'private');
    saveLastModel(subject, { provider: 'local', model: 'private' });
    saveWorkspaceExplorerVisible(subject, false);
    expect(loadLastSessionId(subject)).toBeNull();
    expect(loadLastModel(subject)).toBeNull();
    expect(loadWorkspaceExplorerVisible(subject)).toBe(true);
  }
  expect(localStorage.length).toBe(before);
});
it('does not fail the app when browser storage is blocked', () => {
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new DOMException('Blocked'); });
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new DOMException('Blocked'); });
  vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => { throw new DOMException('Blocked'); });
  expect(loadLastSessionId(a)).toBeNull();
  expect(loadLastModel(a)).toBeNull();
  expect(loadWorkspaceExplorerVisible(a)).toBe(true);
  expect(() => { saveLastSessionId(a, 'x'); saveLastSessionId(a, null); saveLastModel(a, { provider: 'local', model: 'm' }); saveWorkspaceExplorerVisible(a, false); }).not.toThrow();
});
