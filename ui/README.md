# UI (canonical)

UI — основной интерфейс, раздаётся из backend по адресу `/ui/`.

## Сборка

```bash
npm install
npm run build
```

## Важно

`ui/dist/` коммитится намеренно, чтобы backend мог раздавать статический UI без отдельного пайплайна.
