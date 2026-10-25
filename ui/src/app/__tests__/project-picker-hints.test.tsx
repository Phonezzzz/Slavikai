import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it } from 'vitest';
import { BrowserSubjectProvider } from '../browser-subject';
import { ProjectPicker } from '../../features/workspace/project-picker';
const a = 'a'.repeat(64), b = 'b'.repeat(64);
afterEach(() => { cleanup(); localStorage.clear(); });
const picker = (subject: string | null, root: string) => <BrowserSubjectProvider subject={subject}><ProjectPicker sessionId={null} sessionHeader="X-Test" workspaceRoot={root} loading={false} onApplyRoot={() => undefined} onClose={() => undefined} /></BrowserSubjectProvider>;
it('keeps recent directory names private across subject switches and reloads no legacy recents', () => {
  localStorage.setItem('slavik.project-picker.recent-roots', JSON.stringify(['/legacy-private']));
  const h = render(picker(a, '/private-owner-folder'));
  expect(screen.queryByText('/legacy-private')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Open Project' }));
  h.rerender(picker(b, '/member-folder'));
  expect(screen.queryByRole('button', { name: /private-owner-folder/ })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Open Project' }));
  h.rerender(picker(a, '/different-current-root'));
  expect(screen.getByRole('button', { name: /private-owner-folder/ })).toBeTruthy();
  expect(screen.queryByRole('button', { name: /member-folder/ })).toBeNull();
});
