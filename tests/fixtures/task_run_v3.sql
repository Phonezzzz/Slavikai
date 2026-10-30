BEGIN TRANSACTION;
CREATE TABLE run_submissions (
                submission_id TEXT PRIMARY KEY NOT NULL,
                task_run_id TEXT NOT NULL UNIQUE REFERENCES task_runs(task_run_id),
                version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (task_run_id, version)
                REFERENCES run_transitions(task_run_id, version) DEFERRABLE INITIALLY DEFERRED
            );
INSERT INTO "run_submissions" VALUES('87506c84-5a3c-4850-be1a-3e2cb05c474b','a17b0a38-0af1-475e-a0a1-83f225cbcec0',2,'{"attempt_id":"7a61e63c-1c78-4933-8275-617a3c52347e","created_at":"2026-10-28T17:47:22.110303+00:00","failure_code":null,"failure_message":null,"principal_id":"alice","producer_kind":"auto","schema_version":1,"session_id":"session","submission_id":"87506c84-5a3c-4850-be1a-3e2cb05c474b","task_id":"9ccc4d95-33b1-4db1-9627-f176d189a9e9","task_revision":1,"task_run_id":"a17b0a38-0af1-475e-a0a1-83f225cbcec0","text":"producer-v3","version":2}','3ac453e5e3541d4102610d5306637aaf162bd153509553de1691cdddb1f2eed7','2026-10-28T17:47:22.110303+00:00');
CREATE TABLE run_transitions (
        task_run_id TEXT NOT NULL REFERENCES task_runs(task_run_id),
        version INTEGER NOT NULL,
        prior_state TEXT,
        state TEXT NOT NULL,
        authority TEXT NOT NULL,
        reason_code TEXT NOT NULL,
        occurred_at TEXT NOT NULL, submission_id TEXT REFERENCES run_submissions(submission_id),
        PRIMARY KEY (task_run_id, version)
    );
INSERT INTO "run_transitions" VALUES('a17b0a38-0af1-475e-a0a1-83f225cbcec0',0,NULL,'admitted','runtime','request_accepted','2026-10-28T17:47:22.109400+00:00',NULL);
INSERT INTO "run_transitions" VALUES('a17b0a38-0af1-475e-a0a1-83f225cbcec0',1,'admitted','running','runtime','started','2026-10-28T17:47:22.109766+00:00',NULL);
INSERT INTO "run_transitions" VALUES('a17b0a38-0af1-475e-a0a1-83f225cbcec0',2,'running','result_submitted','runtime','submitted','2026-10-28T17:47:22.110303+00:00','87506c84-5a3c-4850-be1a-3e2cb05c474b');
CREATE TABLE task_revisions (
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
INSERT INTO "task_revisions" VALUES('9ccc4d95-33b1-4db1-9627-f176d189a9e9',1,'alice','session','old','old request','auto','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
CREATE TABLE task_runs (
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
INSERT INTO "task_runs" VALUES('a17b0a38-0af1-475e-a0a1-83f225cbcec0','9ccc4d95-33b1-4db1-9627-f176d189a9e9',1,'7a61e63c-1c78-4933-8275-617a3c52347e','result_submitted',2,'2026-10-28T17:47:22.109400+00:00',NULL);
CREATE TRIGGER immutable_transition_update
    BEFORE UPDATE ON run_transitions
    BEGIN SELECT RAISE(ABORT, 'immutable transition'); END;
CREATE TRIGGER immutable_transition_delete
    BEFORE DELETE ON run_transitions
    BEGIN SELECT RAISE(ABORT, 'immutable transition'); END;
CREATE TRIGGER immutable_revision_update
    BEFORE UPDATE ON task_revisions
    BEGIN SELECT RAISE(ABORT, 'immutable revision'); END;
CREATE TRIGGER immutable_revision_delete
    BEFORE DELETE ON task_revisions
    BEGIN SELECT RAISE(ABORT, 'immutable revision'); END;
CREATE TRIGGER immutable_submission_update BEFORE UPDATE ON run_submissions BEGIN SELECT RAISE(ABORT, 'immutable submission'); END;
CREATE TRIGGER immutable_submission_delete BEFORE DELETE ON run_submissions BEGIN SELECT RAISE(ABORT, 'immutable submission'); END;
CREATE TRIGGER submitted_result_binding BEFORE INSERT ON run_transitions
            WHEN (NEW.state = 'result_submitted' AND (
                NEW.submission_id IS NULL OR NOT EXISTS (
                    SELECT 1 FROM run_submissions s
                    WHERE s.submission_id = NEW.submission_id
                    AND s.task_run_id = NEW.task_run_id AND s.version = NEW.version
                    AND s.created_at = NEW.occurred_at
                )
            )) OR (NEW.state != 'result_submitted' AND NEW.submission_id IS NOT NULL)
            BEGIN SELECT RAISE(ABORT, 'submission binding required'); END;
COMMIT;
PRAGMA user_version=3;
