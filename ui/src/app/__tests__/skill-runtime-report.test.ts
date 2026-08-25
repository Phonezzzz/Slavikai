import { describe, expect, it } from 'vitest';

import { parseAutoState, parseMwvReport } from '../session-payload';
import { buildMessageBlocks } from '../../features/messages/block-factory';

describe('engineering skill runtime observability', () => {
  it('parses terminal skill metadata in MWV reports and renders a skill block', () => {
    const report = parseMwvReport({
      route: 'auto',
      skill: {
        status: 'completed',
        skill_id: 'implement',
        version: '1.0.0',
        supporting_skills: [{ skill_id: 'codebase-design', version: '1.0.0' }],
      },
    });

    expect(report?.skill).toEqual({
      status: 'completed',
      skill_id: 'implement',
      version: '1.0.0',
      supporting_skills: [{ skill_id: 'codebase-design', version: '1.0.0' }],
    });

    const blocks = buildMessageBlocks('chat', {
      kind: 'message',
      message: {
        id: 'message-1',
        messageId: 'message-1',
        role: 'assistant',
        content: 'done',
      },
      meta: {
        messageId: 'message-1',
        lane: 'chat',
        traceId: null,
        isFinal: true,
        mwvReport: report,
      },
    });

    expect(blocks[0]).toMatchObject({
      kind: 'skill',
      summary: 'implement@1.0.0 · completed',
    });
  });

  it('preserves skipped skill state in Auto state and rejects non-terminal status', () => {
    const state = parseAutoState({
      run_id: 'auto-1',
      status: 'completed',
      goal: 'chat',
      pool_size: 1,
      started_at: '2026-08-23T00:00:00Z',
      updated_at: '2026-08-23T00:00:01Z',
      skill: {
        status: 'skipped',
        skill_id: null,
        version: null,
        supporting_skills: [],
        reason: 'no_match',
      },
    });

    expect(state?.skill).toEqual({
      status: 'skipped',
      skill_id: null,
      version: null,
      supporting_skills: [],
      reason: 'no_match',
    });
    expect(parseMwvReport({ skill: { status: 'running' } })?.skill).toBeUndefined();
  });

  it('does not render a skill block when no skill was matched or used', () => {
    const report = parseMwvReport({
      route: 'auto',
      skill: {
        status: 'skipped',
        skill_id: null,
        version: null,
        supporting_skills: [],
        reason: 'no_match',
      },
    });

    const blocks = buildMessageBlocks('chat', {
      kind: 'message',
      message: {
        id: 'message-2',
        messageId: 'message-2',
        role: 'assistant',
        content: 'done',
      },
      meta: {
        messageId: 'message-2',
        lane: 'chat',
        traceId: null,
        isFinal: true,
        mwvReport: report,
      },
    });

    expect(blocks.some((block) => block.kind === 'skill')).toBe(false);
  });

  it('renders a skill block when a matched skill failed', () => {
    const report = parseMwvReport({
      route: 'auto',
      skill: {
        status: 'failed',
        skill_id: 'implement',
        version: '1.0.0',
        supporting_skills: [],
      },
    });

    const blocks = buildMessageBlocks('chat', {
      kind: 'message',
      message: {
        id: 'message-3',
        messageId: 'message-3',
        role: 'assistant',
        content: 'done',
      },
      meta: {
        messageId: 'message-3',
        lane: 'chat',
        traceId: null,
        isFinal: true,
        mwvReport: report,
      },
    });

    expect(blocks[0]).toMatchObject({
      kind: 'skill',
      summary: 'implement@1.0.0 · failed',
    });
  });
});
