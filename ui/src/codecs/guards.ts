export const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
};

export const getRecord = (value: unknown, key: string): Record<string, unknown> | null => {
  if (!isRecord(value)) {
    return null;
  }
  const nested = value[key];
  return isRecord(nested) ? nested : null;
};

export const getString = (value: unknown, key: string): string | null => {
  if (!isRecord(value)) {
    return null;
  }
  const nested = value[key];
  return typeof nested === 'string' ? nested : null;
};

export const getNumber = (value: unknown, key: string): number | null => {
  if (!isRecord(value)) {
    return null;
  }
  const nested = value[key];
  return typeof nested === 'number' && Number.isFinite(nested) ? nested : null;
};

export const getBoolean = (value: unknown, key: string): boolean | null => {
  if (!isRecord(value)) {
    return null;
  }
  const nested = value[key];
  return typeof nested === 'boolean' ? nested : null;
};

export const getArray = (value: unknown, key: string): unknown[] | null => {
  if (!isRecord(value)) {
    return null;
  }
  const nested = value[key];
  return Array.isArray(nested) ? nested : null;
};
