# UI Consolidation Inventory (PR-241)

## 1) Runtime Entry Graph

Canonical runtime chain:

1. `ui/src/main.tsx`
2. `ui/src/App.tsx`
3. `ui/src/app/App.tsx`

`ui/src/features/chat/app-shell.tsx` is not imported by this chain and is therefore not runtime-reachable.

## 2) Reachable Set (from entry graph)

Primary reachable modules and feature slices:

- `ui/src/app/App.tsx`
- `ui/src/app/types.ts`
- `ui/src/app/components/ChatArea.tsx`
- `ui/src/app/components/Sidebar.tsx`
- `ui/src/app/components/Settings.tsx`
- `ui/src/app/components/search-modal.tsx`
- `ui/src/app/components/workspace-ide.tsx`
- `ui/src/app/components/workspace-settings-modal.tsx`
- `ui/src/app/components/decision-panel.tsx`
- `ui/src/app/components/plan-panel.tsx`
- `ui/src/app/components/history-sidebar.tsx`
- `ui/src/app/components/ui/dropdown-menu.tsx`
- `ui/src/app/components/ui/utils.ts`
- `ui/src/features/workspace/*` (API, explorer, toolbar, editor pane, assistant panel, helpers, context)

## 3) Subsystem Matrix

| Subsystem | Current implementation | Duplicates found | Runtime canonical |
| --- | --- | --- | --- |
| settings | `ui/src/app/components/Settings.tsx` + `/ui/api/settings` | legacy shell had parallel wiring | `ui/src/app/components/Settings.tsx` |
| models | `ui/src/app/App.tsx` (`/ui/api/models`, `/ui/api/model`) | legacy shell duplicate logic | `ui/src/app/App.tsx` |
| workflow (mode/plan/task) | `ui/src/app/App.tsx` (`/ui/api/mode`, `/ui/api/plan/*`) | legacy shell duplicate logic | `ui/src/app/App.tsx` |
| session/history | `ui/src/app/App.tsx` + `Sidebar`/`history-sidebar` | legacy shell duplicate logic | `ui/src/app/App.tsx` |
| SSE/streaming | `ui/src/app/App.tsx` (`EventSource` stream handling) | legacy shell referenced removed SSE abstraction modules | `ui/src/app/App.tsx` |
| workspace | `ui/src/app/components/workspace-ide.tsx` + `ui/src/features/workspace/*` | none active | current split is canonical |
| attachments | `ui/src/app/App.tsx` + `ChatArea`/canvas components | legacy shell duplicate message handling | `ui/src/app/*` |
| codecs | active parsing lives in `ui/src/app/App.tsx` parser helpers | legacy referenced removed codecs modules | parser helpers in canonical app shell |

## 4) Legacy Candidates

Identified non-canonical legacy surface:

- `ui/src/features/chat/app-shell.tsx`
- Old generated UI-kit components under `ui/src/app/components/ui/*` (except `dropdown-menu.tsx` and `utils.ts`) that are not imported by runtime-reachable modules.

Missing imports inside legacy shell confirmed:

- `../../codecs/settings.codec`
- `../../codecs/session_workflow.codec`
- `../../services/sse/sse_client`
- `../../services/sse/sse_router`
- `./app-shell-utils`

## 5) Decision Log (duplicates -> canonical)

1. **Shell duplication (`app/App.tsx` vs `features/chat/app-shell.tsx`)**
   - Decision: `ui/src/app/App.tsx` is canonical.
   - Action: move legacy shell to archive path `ui/src/legacy/chat/app-shell.tsx`.
   - Rationale: only canonical shell is entry-reachable; legacy had broken imports and no runtime path.

2. **UI-kit dead layer in `ui/src/app/components/ui`**
   - Decision: keep only runtime-used files in active tree (`dropdown-menu.tsx`, `utils.ts`).
   - Action: move non-reachable UI-kit files to `ui/src/legacy/ui-kit/*`.
   - Rationale: remove dead compile surface from active source while preserving artifacts for historical reference.

3. **Typecheck boundary**
   - Decision: typecheck only active source.
   - Action: add `"exclude": ["src/legacy"]` in `ui/tsconfig.json`.
   - Rationale: legacy archive must not affect CI/type gates.

4. **Quality gate**
   - Decision: enforce UI typecheck in default repo checks.
   - Action: add `npm run typecheck` script and include `make ui-type` in `make check`.
   - Rationale: prevent future runtime regressions hidden by `vite build`.
