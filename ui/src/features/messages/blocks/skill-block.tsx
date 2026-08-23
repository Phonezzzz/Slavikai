import { ChevronDown, Workflow } from 'lucide-react';

import type { SkillRunReportUi } from '../../../app/types';
import { DetailsPanel } from './details-panel';

type SkillBlockProps = {
  summary: string;
  skill: SkillRunReportUi;
  open: boolean;
  onToggle: () => void;
};

export function SkillBlock({ summary, skill, open, onToggle }: SkillBlockProps) {
  return (
    <div className="message-diagnostic message-diagnostic--skill">
      <div className="message-diagnostic-summary">
        <div className="message-diagnostic-left">
          <Workflow className="h-3.5 w-3.5" />
          <span>Skill</span>
          <span className="message-diagnostic-divider" />
          <span className="message-diagnostic-text">{summary}</span>
        </div>
        <button
          type="button"
          onClick={onToggle}
          className="message-diagnostic-toggle"
          aria-expanded={open}
          aria-label={open ? 'Hide skill details' : 'Show skill details'}
        >
          <span>Details</span>
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>
      <DetailsPanel open={open}>
        <div className="message-details-stack">
          <div className="message-diagnostic-entry">
            <div className="message-diagnostic-entry-title">Skill run</div>
            <pre className="message-diagnostic-pre">{JSON.stringify(skill, null, 2)}</pre>
          </div>
        </div>
      </DetailsPanel>
    </div>
  );
}
