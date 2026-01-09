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
- `entrypoints`: non-empty list of tool or flow identifiers
- `patterns`: non-empty list of matching phrases (substring match)
- `requires`: list of dependencies (can be empty)
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

Runtime reads only `_generated/skills.manifest.json` and builds a pattern index.
File system scanning is not allowed at runtime.

Dev mode (optional): set `SKILLS_DEV_MODE=1` to rebuild the manifest before loading.

## Candidates

When the agent sees an unknown tool/code request or repeated tool errors, it creates
candidate drafts under `_candidates/`. Candidates are not active until manually
promoted into `skills/<id>/skill.md` and the manifest is rebuilt.
