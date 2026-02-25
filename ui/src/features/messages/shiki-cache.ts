import type { MessageRenderContext } from './types';
import type { LanguageRegistration, ThemeRegistrationAny } from 'shiki/types';

type HighlightRequest = {
  context: MessageRenderContext;
  code: string;
  language: string | null;
};

type HighlighterLike = {
  codeToHtml: (code: string, options: { lang: string; theme: string }) => string;
};

type LanguageDefinition = LanguageRegistration | LanguageRegistration[];

type LanguageKey =
  | 'bash'
  | 'diff'
  | 'json'
  | 'yaml'
  | 'markdown'
  | 'python'
  | 'javascript'
  | 'typescript'
  | 'tsx'
  | 'jsx'
  | 'html'
  | 'css';

type ThemeKey = 'github-dark-dimmed' | 'one-dark-pro';

const FALLBACK_LANGUAGE = 'text';
const FALLBACK_THEME: ThemeKey = 'github-dark-dimmed';

const THEME_BY_CONTEXT: Record<MessageRenderContext, ThemeKey> = {
  chat: 'github-dark-dimmed',
  workspace: 'one-dark-pro',
};

const ALLOWED_HIGHLIGHT_LANGS: readonly LanguageKey[] = [
  'bash',
  'diff',
  'json',
  'yaml',
  'markdown',
  'python',
  'javascript',
  'typescript',
  'tsx',
  'jsx',
  'html',
  'css',
] as const;

const AVAILABLE_THEMES: readonly ThemeKey[] = ['github-dark-dimmed', 'one-dark-pro'] as const;

const ALLOWED_LANGUAGE_SET = new Set<string>([
  ...ALLOWED_HIGHLIGHT_LANGS,
  FALLBACK_LANGUAGE,
]);

const LANGUAGE_ALIASES: Record<string, string> = {
  sh: 'bash',
  shell: 'bash',
  shellscript: 'bash',
  yml: 'yaml',
  md: 'markdown',
  py: 'python',
  js: 'javascript',
  ts: 'typescript',
  txt: 'text',
};

const highlightCache = new Map<string, string>();
let highlighterPromise: Promise<HighlighterLike> | null = null;

const LANGUAGE_LOADERS: Record<LanguageKey, () => Promise<LanguageDefinition>> = {
  bash: () => import('@shikijs/langs/bash').then((module) => module.default),
  diff: () => import('@shikijs/langs/diff').then((module) => module.default),
  json: () => import('@shikijs/langs/json').then((module) => module.default),
  yaml: () => import('@shikijs/langs/yaml').then((module) => module.default),
  markdown: () => import('@shikijs/langs/markdown').then((module) => module.default),
  python: () => import('@shikijs/langs/python').then((module) => module.default),
  javascript: () => import('@shikijs/langs/javascript').then((module) => module.default),
  typescript: () => import('@shikijs/langs/typescript').then((module) => module.default),
  tsx: () => import('@shikijs/langs/tsx').then((module) => module.default),
  jsx: () => import('@shikijs/langs/jsx').then((module) => module.default),
  html: () => import('@shikijs/langs/html').then((module) => module.default),
  css: () => import('@shikijs/langs/css').then((module) => module.default),
};

const THEME_LOADERS: Record<ThemeKey, () => Promise<ThemeRegistrationAny>> = {
  'github-dark-dimmed': () =>
    import('@shikijs/themes/github-dark-dimmed').then((module) => module.default),
  'one-dark-pro': () =>
    import('@shikijs/themes/one-dark-pro').then((module) => module.default),
};

const stableHash = (value: string): string => {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16);
};

const normalizeLanguage = (language: string | null): string => {
  if (!language) {
    return FALLBACK_LANGUAGE;
  }
  const normalized = language.trim().toLowerCase();
  if (!normalized) {
    return FALLBACK_LANGUAGE;
  }
  const aliased = LANGUAGE_ALIASES[normalized] ?? normalized;
  if (!ALLOWED_LANGUAGE_SET.has(aliased)) {
    return FALLBACK_LANGUAGE;
  }
  return aliased;
};

const normalizeTheme = (context: MessageRenderContext): ThemeKey => {
  const theme = THEME_BY_CONTEXT[context];
  if (theme && AVAILABLE_THEMES.includes(theme)) {
    return theme;
  }
  return FALLBACK_THEME;
};

const loadAllowedLanguages = async (): Promise<LanguageRegistration[]> => {
  const loaded = await Promise.all(ALLOWED_HIGHLIGHT_LANGS.map((key) => LANGUAGE_LOADERS[key]()));
  const flattened: LanguageRegistration[] = [];
  loaded.forEach((entry) => {
    if (Array.isArray(entry)) {
      flattened.push(...entry);
      return;
    }
    flattened.push(entry);
  });
  return flattened;
};

const loadAllowedThemes = async (): Promise<ThemeRegistrationAny[]> => {
  return Promise.all(AVAILABLE_THEMES.map((key) => THEME_LOADERS[key]()));
};

const createViaBundledApi = async (): Promise<HighlighterLike | null> => {
  try {
    const [coreModule, onigurumaModule] = await Promise.all([
      import('shiki/core'),
      import('shiki/engine/oniguruma'),
    ]);

    if (typeof coreModule.createBundledHighlighter !== 'function') {
      return null;
    }

    const factory = coreModule.createBundledHighlighter({
      langs: LANGUAGE_LOADERS,
      themes: THEME_LOADERS,
      engine: () => onigurumaModule.createOnigurumaEngine(),
    });

    return await factory({
      langs: [...ALLOWED_HIGHLIGHT_LANGS],
      themes: [...AVAILABLE_THEMES],
    });
  } catch {
    return null;
  }
};

const createViaCoreApi = async (): Promise<HighlighterLike | null> => {
  try {
    const [coreModule, onigurumaModule, langs, themes] = await Promise.all([
      import('shiki/core'),
      import('shiki/engine/oniguruma'),
      loadAllowedLanguages(),
      loadAllowedThemes(),
    ]);
    if (typeof coreModule.createHighlighterCore !== 'function') {
      return null;
    }
    return await coreModule.createHighlighterCore({
      langs,
      themes,
      engine: onigurumaModule.createOnigurumaEngine(),
    });
  } catch {
    return null;
  }
};

const createHighlighter = async (): Promise<HighlighterLike> => {
  const bundled = await createViaBundledApi();
  if (bundled) {
    return bundled;
  }
  const core = await createViaCoreApi();
  if (core) {
    return core;
  }
  throw new Error('Failed to initialize Shiki highlighter from exported API.');
};

const getHighlighter = (): Promise<HighlighterLike> => {
  if (highlighterPromise) {
    return highlighterPromise;
  }
  highlighterPromise = createHighlighter();
  return highlighterPromise;
};

export const highlightCodeWithCache = async ({ context, code, language }: HighlightRequest): Promise<string> => {
  const theme = normalizeTheme(context);
  const normalizedLanguage = normalizeLanguage(language);
  const cacheKey = `${theme}:${normalizedLanguage}:${stableHash(code)}`;
  const cached = highlightCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const highlighter = await getHighlighter();
  try {
    const html = highlighter.codeToHtml(code, {
      lang: normalizedLanguage,
      theme,
    });
    highlightCache.set(cacheKey, html);
    return html;
  } catch {
    const html = highlighter.codeToHtml(code, {
      lang: FALLBACK_LANGUAGE,
      theme: FALLBACK_THEME,
    });
    highlightCache.set(cacheKey, html);
    return html;
  }
};
