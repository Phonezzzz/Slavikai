export type ComposerAttachment = {
  name: string;
  mime: string;
  content: string;
};

export const MAX_COMPOSER_ATTACHMENTS = 8;
export const MAX_ATTACHMENT_CONTENT_CHARS = 80_000;

const TEXT_FILE_PATTERN =
  /\.(txt|md|markdown|json|yaml|yml|toml|csv|log|py|ts|tsx|js|jsx|css|scss|html|xml|sh|bash|zsh|ini|cfg|conf|sql|env)$/i;
const IMAGE_SCALE_STEPS = [1, 0.85, 0.7, 0.55, 0.4, 0.3, 0.22];
const IMAGE_QUALITY_STEPS = [0.92, 0.82, 0.7, 0.56, 0.44, 0.34];

const fallbackNameForMime = (mime: string): string => {
  if (mime === 'image/jpeg') {
    return 'pasted-image.jpg';
  }
  if (mime === 'image/webp') {
    return 'pasted-image.webp';
  }
  if (mime === 'image/gif') {
    return 'pasted-image.gif';
  }
  return 'pasted-image.png';
};

const normalizeFileName = (name: string | null | undefined, fallback: string): string => {
  const normalized = (name ?? '').trim();
  return normalized || fallback;
};

const canReadAsText = (file: File): boolean => {
  if (file.type.startsWith('text/')) {
    return true;
  }
  return TEXT_FILE_PATTERN.test(file.name);
};

const ensureWithinAttachmentLimit = (content: string, fileName: string): string => {
  if (content.length > MAX_ATTACHMENT_CONTENT_CHARS) {
    throw new Error(
      `Файл ${fileName} слишком большой для одного вложения (${MAX_ATTACHMENT_CONTENT_CHARS} символов).`,
    );
  }
  return content;
};

const readFileAsDataUrl = async (file: Blob): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
        return;
      }
      reject(new Error('Не удалось прочитать бинарное вложение.'));
    };
    reader.onerror = () => reject(new Error('Не удалось прочитать бинарное вложение.'));
    reader.readAsDataURL(file);
  });

const loadImage = async (file: Blob): Promise<HTMLImageElement> => {
  const objectUrl = URL.createObjectURL(file);
  try {
    return await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error('Не удалось декодировать изображение.'));
      image.src = objectUrl;
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
};

const compressImageToDataUrl = async (file: File): Promise<{ dataUrl: string; mime: string }> => {
  const originalDataUrl = await readFileAsDataUrl(file);
  if (originalDataUrl.length <= MAX_ATTACHMENT_CONTENT_CHARS) {
    return {
      dataUrl: originalDataUrl,
      mime: file.type || 'image/png',
    };
  }

  const image = await loadImage(file);
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas недоступен для сжатия изображения.');
  }

  const mimeCandidates =
    file.type === 'image/jpeg' || file.type === 'image/webp'
      ? [file.type, 'image/jpeg']
      : ['image/jpeg'];

  for (const scale of IMAGE_SCALE_STEPS) {
    const width = Math.max(1, Math.round(image.naturalWidth * scale));
    const height = Math.max(1, Math.round(image.naturalHeight * scale));
    canvas.width = width;
    canvas.height = height;
    context.clearRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);

    for (const mime of mimeCandidates) {
      for (const quality of IMAGE_QUALITY_STEPS) {
        const dataUrl = canvas.toDataURL(mime, quality);
        if (dataUrl.length <= MAX_ATTACHMENT_CONTENT_CHARS) {
          return { dataUrl, mime };
        }
      }
    }
  }

  throw new Error(
    `Изображение ${normalizeFileName(file.name, fallbackNameForMime(file.type || 'image/png'))} слишком большое даже после сжатия.`,
  );
};

const buildBinaryAttachmentNote = (file: File, mime: string): string => {
  return [
    '[binary attachment]',
    `name: ${normalizeFileName(file.name, 'attachment.bin')}`,
    `mime: ${mime}`,
    `size_bytes: ${file.size}`,
  ].join('\n');
};

export const createComposerAttachmentId = (prefix = 'attachment'): string => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

export const buildPastedTextAttachment = (text: string): ComposerAttachment => {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  return {
    name: `pasted-${stamp}.txt`,
    mime: 'text/plain',
    content: ensureWithinAttachmentLimit(text, `pasted-${stamp}.txt`),
  };
};

export const readComposerAttachmentFromFile = async (file: File): Promise<ComposerAttachment> => {
  const normalizedName = normalizeFileName(file.name, fallbackNameForMime(file.type || 'application/octet-stream'));

  if (file.type.startsWith('image/')) {
    const compressed = await compressImageToDataUrl(file);
    return {
      name: normalizedName,
      mime: compressed.mime,
      content: compressed.dataUrl,
    };
  }

  if (canReadAsText(file)) {
    const text = await file.text();
    return {
      name: normalizedName,
      mime: file.type || 'text/plain',
      content: ensureWithinAttachmentLimit(text, normalizedName),
    };
  }

  const mime = file.type || 'application/octet-stream';
  return {
    name: normalizedName,
    mime,
    content: ensureWithinAttachmentLimit(buildBinaryAttachmentNote(file, mime), normalizedName),
  };
};

