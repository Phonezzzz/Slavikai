import { cleanup, fireEvent, render } from '@testing-library/react';
import { useRef } from 'react';
import { afterEach, describe, expect, it } from 'vitest';

import { useFocusTrap } from '../use-focus-trap';

function TrapHarness({ active }: { active: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useFocusTrap(active, ref);
  return (
    <div ref={ref}>
      <button type="button">first</button>
      <button type="button">second</button>
      <a href="#x">link</a>
    </div>
  );
}

afterEach(() => {
  cleanup();
});

describe('useFocusTrap', () => {
  it('moves focus into the container when activated', () => {
    render(<TrapHarness active />);
    expect(document.activeElement?.textContent).toBe('first');
  });

  it('wraps Tab from the last focusable back to the first', () => {
    const { container } = render(<TrapHarness active />);
    const buttons = Array.from(container.querySelectorAll<HTMLElement>('button, a'));
    const last = buttons[buttons.length - 1];
    last.focus();
    fireEvent.keyDown(document, { key: 'Tab' });
    expect(document.activeElement).toBe(buttons[0]);
  });

  it('wraps Shift+Tab from the first focusable back to the last', () => {
    const { container } = render(<TrapHarness active />);
    const buttons = Array.from(container.querySelectorAll<HTMLElement>('button, a'));
    const last = buttons[buttons.length - 1];
    buttons[0].focus();
    fireEvent.keyDown(document, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(last);
  });

  it('does not trap focus when inactive', () => {
    render(<TrapHarness active={false} />);
    expect(document.activeElement).toBe(document.body);
  });
});
