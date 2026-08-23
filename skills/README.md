# Skills

This directory stores skill definitions and the generated manifest used at runtime.

## Layout

```
skills/
  README.md
  _schema/skill.schema.json
  _generated/skills.manifest.json
  _generated/skills.bundle.json            # optional
  _candidates/                             # auto-created drafts (not active)
  tools/
    build_manifest.py
    lint_skills.py
  <skill-id>/
    skill.md
    tests/                                 # optional
```

## skill.md format

Each `skills/<skill-id>/skill.md` must start with YAML front matter. Required fields:

- `id`: unique string
- `version`: semver (`MAJOR.MINOR.PATCH`)
- `title`: human readable title
- `entrypoints`: tool or flow identifiers; non-empty for workflow skills
- `patterns`: matching phrases (substring match); non-empty for workflow skills
- `requires`: list of dependencies (can be empty)
- `dependencies`: supporting skill ids to inject before this skill (can be empty)
- `supporting`: `true` for dependency-only skills, otherwise `false`
- `risk`: `low | medium | high`
- `tests`: list of test paths (can be empty)

Optional fields:

- `deprecated`: boolean
- `replaced_by`: string (required if `deprecated: true`)

## Tools

- `python skills/tools/lint_skills.py`
- `python skills/tools/build_manifest.py`
- `python skills/tools/build_manifest.py --check`

## Runtime loading

Runtime reads only `_generated/skills.manifest.json`; manifest version 2 embeds the complete
Markdown body as `instructions`, dependency metadata, and a pattern index. A matched workflow
skill is resolved with its supporting dependencies and injected into that run's system context.
Supporting skills cannot be matched directly. Skill instructions never add tools or bypass
`ToolGateway`, approvals, sandbox policy, safe mode, or verifier. Front-matter `entrypoints` remain
descriptive metadata; the active ToolRegistry and ToolGateway are the only execution authority.
File system scanning is not allowed at runtime.

Dev mode (optional): set `SKILLS_DEV_MODE=1` to rebuild the manifest before loading.

## Candidates

When the agent sees an unknown tool/code request or repeated tool errors, it creates
candidate drafts under `_candidates/`. Candidates are not active until manually
promoted into `skills/<id>/skill.md` and the manifest is rebuilt.

Operational rules:
- Candidates have a TTL and are archived when expired.
- The directory is capped (older drafts are archived).
- A cooldown by fingerprint prevents write storms.
Defaults can be overridden via env:
`SKILLS_CANDIDATE_MAX`, `SKILLS_CANDIDATE_TTL_DAYS`, `SKILLS_CANDIDATE_COOLDOWN_SECONDS`.

## Lifecycle (minimal)

1. Request arrives → routing checks triggers + `SkillIndex`.
2. If a skill matches in Auto → Auto v1 executes it with per-run instructions; the MWV report
   records the selected skill and terminal status.
   - Deprecated or ambiguous skill → request is blocked with an explicit message.
3. Verifier (`scripts/check.sh`) is the only success gate.
4. If no skill matches for MWV‑type request → candidate draft is created.
5. `_candidates` never participate in routing until promoted.

## Add a new skill (minimal)

1. Create `skills/<id>/skill.md` with valid front matter.
2. Run:
   - `python skills/tools/lint_skills.py`
   - `python skills/tools/build_manifest.py`
3. Ensure `_generated/skills.manifest.json` is updated and committed.
4. (Optional) add tests under `skills/<id>/tests/`.
