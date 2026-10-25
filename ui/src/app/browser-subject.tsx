import { createContext, useContext, type ReactNode } from 'react';

const BrowserSubject = createContext<string | null>(null);
export const verifiedBrowserSubject = (value: unknown): string | null => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value) ? value : null;
export const useBrowserSubject = () => useContext(BrowserSubject);

export function BrowserSubjectProvider({ subject, children }: { subject: string | null; children: ReactNode }) {
  const verified = verifiedBrowserSubject(subject);
  return <BrowserSubject.Provider key={verified ?? 'no-persistence'} value={verified}>{children}</BrowserSubject.Provider>;
}
