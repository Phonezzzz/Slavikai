import { useCallback, useEffect, useState } from 'react';

type DesktopScope = {
  tool?: unknown;
  action?: unknown;
  target_pattern?: unknown;
  command_class?: unknown;
  risk_class?: unknown;
};

type DesktopRule = {
  rule_id: string;
  effect: 'allow' | 'ask' | 'deny';
  scope: DesktopScope;
  description: string;
};

const parseRules = (payload: unknown): DesktopRule[] => {
  const raw = (payload as { rules?: unknown } | null)?.rules;
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.flatMap((item) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      return [];
    }
    const record = item as Record<string, unknown>;
    const ruleId = record.rule_id;
    const effect = record.effect;
    const scope = record.scope;
    if (
      typeof ruleId !== 'string'
      || !['allow', 'ask', 'deny'].includes(String(effect))
      || !scope
      || typeof scope !== 'object'
      || Array.isArray(scope)
    ) {
      return [];
    }
    return [{
      rule_id: ruleId,
      effect: effect as DesktopRule['effect'],
      scope: scope as DesktopScope,
      description: typeof record.description === 'string' ? record.description : '',
    }];
  });
};

export function DesktopApprovalsPanel() {
  const [rules, setRules] = useState<DesktopRule[]>([]);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    const response = await fetch('/ui/api/desktop/approvals');
    const payload = await response.json() as unknown;
    if (!response.ok) {
      throw new Error('Failed to load Desktop approvals.');
    }
    setRules(parseRules(payload));
  }, []);

  useEffect(() => {
    void reload().catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : 'Failed to load Desktop approvals.');
    });
  }, [reload]);

  const updateEffect = async (rule: DesktopRule) => {
    const nextEffect = rule.effect === 'allow' ? 'deny' : 'allow';
    const response = await fetch(`/ui/api/desktop/approvals/${encodeURIComponent(rule.rule_id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ effect: nextEffect }),
    });
    if (!response.ok) {
      throw new Error('Failed to update Desktop approval.');
    }
    await reload();
  };

  const remove = async (ruleId: string) => {
    const response = await fetch(`/ui/api/desktop/approvals/${encodeURIComponent(ruleId)}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('Failed to remove Desktop approval.');
    }
    await reload();
  };

  return (
    <div className="mt-2 rounded-md border border-[#26262e] bg-[#0d0d11] p-2 text-[11px]">
      <div className="flex items-center justify-between">
        <span className="font-medium text-[#c7c7d0]">Persistent Desktop approvals</span>
        <button type="button" onClick={() => void reload()} className="text-[#8f8f99] hover:text-white">
          Refresh
        </button>
      </div>
      {rules.length === 0 ? <div className="mt-2 text-[#777782]">No persistent rules.</div> : null}
      {rules.map((rule) => {
        const target = typeof rule.scope.target_pattern === 'string'
          ? rule.scope.target_pattern
          : 'no path scope';
        return (
          <div key={rule.rule_id} className="mt-2 rounded border border-[#23232a] p-2">
            <div className="truncate text-[#bcbcc6]">
              {String(rule.scope.tool ?? '?')}:{String(rule.scope.action ?? '?')} → {target}
            </div>
            <div className="mt-1 flex items-center gap-2">
              <button
                type="button"
                onClick={() => void updateEffect(rule).catch((reason: unknown) => {
                  setError(reason instanceof Error ? reason.message : 'Update failed.');
                })}
                className="rounded border border-[#34343d] px-2 py-0.5 text-[#a8a8b2]"
              >
                {rule.effect}
              </button>
              <button
                type="button"
                onClick={() => void remove(rule.rule_id).catch((reason: unknown) => {
                  setError(reason instanceof Error ? reason.message : 'Delete failed.');
                })}
                className="text-rose-300"
              >
                Remove
              </button>
            </div>
          </div>
        );
      })}
      {error ? <div className="mt-2 text-rose-300">{error}</div> : null}
    </div>
  );
}
