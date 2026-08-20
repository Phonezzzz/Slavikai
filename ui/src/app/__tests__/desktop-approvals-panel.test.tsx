import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render } from '@testing-library/react';

import { DesktopApprovalsPanel } from '../../features/desktop/desktop-approvals-panel';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('DesktopApprovalsPanel', () => {
  it('shows the exact scope and current effect of a persistent rule', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
      ok: true,
      rules: [{
        rule_id: 'rule-1',
        effect: 'allow',
        description: 'one exact download',
        scope: {
          tool: 'desktop_file_delete',
          action: 'delete',
          target_pattern: '/home/user/Downloads/one.iso',
          risk_class: 'destructive',
          execution_target: 'desktop',
        },
      }],
    }), { status: 200 })));

    const { container } = render(<DesktopApprovalsPanel />);

    await vi.waitFor(() => {
      expect(container.textContent).toContain('desktop_file_delete:delete');
      expect(container.textContent).toContain('/home/user/Downloads/one.iso');
      expect(container.textContent).toContain('allow');
    });
  });
});
