CREATE TABLE IF NOT EXISTS task_revisions (
                    task_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision = 1),
                    principal_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_key TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    PRIMARY KEY (task_id, revision),
                    UNIQUE (principal_id, request_key)
                );
CREATE TABLE IF NOT EXISTS task_runs (
                    task_run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    task_revision INTEGER NOT NULL,
                    attempt_id TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 0),
                    created_at TEXT NOT NULL,
                    terminal_at TEXT,
                    FOREIGN KEY (task_id, task_revision)
                    REFERENCES task_revisions(task_id, revision)
                );
CREATE TABLE IF NOT EXISTS run_transitions (
                    task_run_id TEXT NOT NULL REFERENCES task_runs(task_run_id),
                    version INTEGER NOT NULL,
                    prior_state TEXT,
                    state TEXT NOT NULL,
                    authority TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    PRIMARY KEY (task_run_id, version)
                );
CREATE TRIGGER IF NOT EXISTS immutable_transition_update
                BEFORE UPDATE ON run_transitions
                BEGIN SELECT RAISE(ABORT, 'immutable transition'); END;
CREATE TRIGGER IF NOT EXISTS immutable_transition_delete
                BEFORE DELETE ON run_transitions
                BEGIN SELECT RAISE(ABORT, 'immutable transition'); END;
CREATE TRIGGER IF NOT EXISTS immutable_revision_update
                BEFORE UPDATE ON task_revisions
                BEGIN SELECT RAISE(ABORT, 'immutable revision'); END;
CREATE TRIGGER IF NOT EXISTS immutable_revision_delete
                BEFORE DELETE ON task_revisions
                BEGIN SELECT RAISE(ABORT, 'immutable revision'); END;
PRAGMA user_version = 2;
