# ADR-0013 — Owner Host Bridge trust model: AF_UNIX, non-root, bounded capabilities

- **Status:** accepted (E03-T01; owner decisions recorded 2026-11-23)
- **Date:** 2026-11-23
- **Decision owner:** SlavikAI product/architecture owner
- **Affected domains:** host execution boundary, ToolGateway/DesktopPolicy,
  desktop tools, process execution, audit/effect log, systemd deployment

## Context and problem

Today SlavikAI reaches the owner's host **in-process**: the server runs as the
owner's UID and desktop tools execute directly in that process
(`tools/desktop_tools.py` — file search/read/write/transfer/delete, archive
extract, `DesktopShellTool`; `tools/desktop_system_tools.py` — system info and
launches). Path safety is enforced by `core/desktop_security.py`
(`DesktopPathSecurity`: normal/ask/deny classification, protected prefixes,
credential/secret-file denial, mount-topology checks), command safety by
`shared/command_safety.py` (`classify_command`: `privilege_escalation`,
`disk_boot`, `shell_indirection` denied; package/service/process management
must use typed capabilities), and subprocess environment by an allowlist
(`_desktop_subprocess_env`). Policy decisions (ask/deny/grants per
`subject_principal_id`) live in `core/desktop_policy.py` behind the single
execution gate `core/tool_gateway.py`. The sandbox lane is separate
(`shared/sandbox.py`, `tools/filesystem_tool.py`, container backend via
rootless podman in `core/container_computer_backend.py`).

There is **no IPC layer**: no `AF_UNIX`, no `SO_PEERCRED`, no socket code
anywhere in `core/`, `tools/`, `shared/`, `server/`. Consequences of the
in-process model:

- Any code execution inside the server process (model-driven tool bug,
  compromised dependency, prompt-injection-driven tool misuse) inherits the
  full authority of the owner's UID with only in-process checks between it
  and the host.
- Host effects are not attributable to a bounded, independently hardened
  component; the server cannot be sandboxed (systemd hardening, restricted
  filesystem view) without breaking desktop tools.
- There is no transport-level identity: nothing distinguishes "the SlavikAI
  server" from any other process of the same UID at the host boundary.

Epic E03 introduces an **Owner Host Bridge** (Desktop Commander): a separate
daemon the server talks to over a local socket, so host authority becomes a
bounded, auditable capability instead of ambient process authority. This ADR
fixes the trust model **before** any daemon code (E03-T02+), per the ticket
rule "Не начинать daemon до решения trust model".

## Decision

1. **Transport: AF_UNIX stream socket, no TCP.** The bridge listens only on a
   Unix socket in a private directory (mode `0700`, socket mode `0600`).
   World-writable or group-writable socket is impossible by construction
   (E03-T02 negative case). No network listener is opened by the daemon.
   Socket location: `$XDG_RUNTIME_DIR/slavikai/host-bridge.sock`
   (owner decision 2026-11-23). There is no persistent fallback in v1:
   if `XDG_RUNTIME_DIR` is unavailable the daemon does not start
   (fail-closed).

2. **Non-root v1.** The daemon runs as the owner's UID under a systemd
   **user** service. A root bridge is explicitly out of scope for v1; any
   future root variant requires a separate owner gate and a new ADR
   (ticket owner gate). The daemon never invokes `sudo`, `pkexec`, or setuid
   helpers.

3. **Transport principal: SO_PEERCRED.** On every connection the daemon reads
   peer credentials (`SO_PEERCRED`) and accepts only the expected UID (the
   owner's UID, equal to the daemon's own UID). Any other local user gets a
   deterministic refusal before any protocol bytes are processed. This is
   the only authentication the bridge performs; it authenticates *a local
   process of the owner*, not a human user.

4. **Application principal stays in SlavikAI.** Requests carry the SlavikAI
   principal (`principal_id` from the verified Cloudflare Access identity,
   `server/http/common/auth.py`) and a run ID for attribution and audit
   (E03-T08), but the bridge does **not** authorize users. Whether a run may
   use the `owner_host` target at all — owner vs member vs automation lane —
   is decided by ToolGateway/DesktopPolicy before the request is sent.
   **v1: only owner-role runs may target `owner_host`**; member and
   automation (`/v1` bearer) lanes are excluded (owner decision
   2026-11-23).

   Run authorization is **not** taken from request JSON: `principal_id`
   and run ID in a request are attribution/audit data only, never
   authorization input. The right of a connection to act for a specific
   owner-run must be confirmed by the trusted runtime independently of
   those fields (owner decision 2026-11-23). The mechanism (for example
   binding the peer process to the supervised SlavikAI server, or
   short-lived per-run tokens issued over the authenticated connection)
   is a T02/T03 design point inside this constraint; a request that
   merely *claims* an owner principal or run ID gains nothing.

5. **The bridge is an execution backend, not a policy engine.** It plugs in
   as `execution_target="owner_host"` next to the existing `desktop` and
   sandbox/container targets (`core/desktop_policy.py`,
   `core/computer_backend.py`). ToolGateway remains the single policy gate;
   DesktopPolicyRuntime rules (ask/deny/scoped grants, deny/ask over grants —
   E02-T09 semantics) apply unchanged. Bridge-side checks are **hard safety
   bounds**, not policy: allowed roots, forbidden operations, frame limits,
   timeouts. The bridge never asks the user anything; a request is either
   inside the bounds and executes, or is rejected deterministically with a
   typed error. No second policy engine is created (consistent with E02
   decisions).

6. **Capabilities are enumerated, versioned, and mapped to tickets.** The
   daemon exposes only the capabilities below; unknown capability or unknown
   protocol version is a deterministic rejection (E03-T03). Unknown request
   fields never become command arguments.

   | Capability | Ticket | Notes |
   | --- | --- | --- |
   | `bridge.ping`, `bridge.version`, `bridge.capabilities` | T02 | lifecycle/auth proof, no dangerous operations |
   | (protocol validation: versioned schema, frame limit, request ID, timeout) | T03 | applies to every request |
   | `host.fs.stat/list/read/write/patch/move/delete/search` | T04 | only inside allowed roots; reuses `DesktopPathSecurity` internals |
   | `host.process.run/poll/kill` | T05 | approved command profiles only (see Decision §10a); no network for spawned processes in v1; stable process ID; stdout/stderr/exit/timing |
   | run-scoped STOP (kill only processes of the same run) | T06 | process refs bound to run ID |
   | `host.diag.service_status/process_list/disk/host` (read-only) | T07 | `service.restart` does not exist in the registry |
   | structured effect records returned per request | T08 | SlavikAI persists request metadata + effect + observation |
   | systemd hardening profile | T09 | `NoNewPrivileges`, restricted FS view, `PrivateDevices`, address-family limits, explicit writable paths; verified by `systemd-analyze security` + runtime tests |
   | `desktop.observe` / `desktop.control` (split) | T10 | no GUI capability in v1; any GUI access is a separate owner decision in T10 |

7. **Assets** (what the bridge must protect):

   - Owner files inside allowed roots — integrity and confidentiality.
   - Credential material on the host: SSH/GPG keys, browser session stores
     (the owner's real browser profiles and the SlavikAI persistent browser
     profiles under `~/.local/state/slavikai/`), ChatGPT plan grants,
     `.env` files, API keys. These stay **outside** bridge reach even though
     the daemon shares the owner's UID — enforced by root bounds and
     secret-file denial, not by UID separation.
   - Host integrity: no persistence mechanisms (systemd units, cron,
     autostart entries, shell rc files) may be created or modified through
     the bridge.
   - Availability: bounded frames, timeouts, output caps, and per-run
     process ownership prevent runaway processes and disk exhaustion.
   - Audit trail integrity: every effect is attributable to principal + run
     (E03-T08); stdout secrets are redacted, raw environment is never logged.

8. **Attacker classes** (who the bounds must hold against):

   1. **Other local users** on the host (including sandboxed apps under a
      different UID) — stopped by socket permissions + `SO_PEERCRED`.
   2. **The model as confused deputy** — prompt injection via web content,
      files, or tool results steering the agent into hostile tool calls.
      Stopped by ToolGateway/DesktopPolicy (ask/deny) **and** by bridge hard
      bounds (roots, forbidden operations) even if policy is bypassed.
   3. **Malicious web content** driving the E02 browser agent and pivoting
      from browser capabilities to host capabilities — same bounds as (2);
      browser and host capabilities remain separate grants.
   4. **Compromised bridge client** (the SlavikAI server process itself) —
      the daemon still enforces roots/forbidden operations/frame limits, so
      server compromise does not silently become arbitrary host compromise.
   5. **Malware running as the owner's UID** — `SO_PEERCRED` cannot
      distinguish it from the real client. Mitigated only partially (private
      socket directory, capability scoping, audit); residual risk is
      accepted and documented: same-UID malware already owns the owner's
      files without the bridge.
   6. **Supply-chain payloads** — a malicious binary/script placed inside an
      allowed root and then executed via `host.process.run`. Mitigation is
      an owner decision (exec restrictions — see Open questions).

9. **Allowed roots (owner decision 2026-11-23).** The daemon resolves
   every path against this list after canonicalization; `../`, absolute
   escape, symlink escape, and destination escape are rejected (E03-T04
   negatives).

   - **Default and only built-in root:** `~/.local/share/slavikai/host-workspace`
     — read/write (also the E2E test root for E03-T11).
   - The owner may explicitly allow **specific subdirectories** under
     `~/Project`, `~/Documents`, and `~/Downloads`, each with its own
     access mode (read-only or read/write). The parent directories
     themselves are never roots.
   - **Write-excluded even inside an allowed root:** SlavikAI checkouts,
     the bridge's own launch files (systemd unit, daemon config), the
     policy store, and runtime state. A root grant never makes the
     agent's own code, policy, or state writable through the bridge.
   - Secret-file denial applies **on top of** the allowlist (§10): an
     allowed root does not make credential material readable.
   - Explicitly **never** roots: `~/.ssh`, `~/.gnupg`, `~/.config` (real
     browser profiles), `~/.local/state/slavikai` (browser profiles, leases,
     ChatGPT grants — already private runtime paths per E02-T07), `/etc`,
     `/usr`, `/root`, the home directory as a whole, and any ancestor of the
     daemon's own config/policy store. Note: `DesktopPathSecurity`
     classifies `/etc` as `ask`; the bridge bound is stricter — `/etc`
     is a hard daemon-side deny, recorded explicitly so the in-process
     classification cannot leak into bridge behavior.

10. **Forbidden operations (owner decision 2026-11-23).** Enforced
    daemon-side regardless of client requests:

    - Privilege escalation: `sudo`, `pkexec`, setuid/setgid binaries,
      capability manipulation.
    - Shell interpretation: no `/bin/sh -c`, no shell strings; process API
      is argv-only (E03-T05). Environment is an allowlist (starting point:
      the existing `_desktop_subprocess_env` set), never full inheritance.
    - Writes outside allowed roots; mount/bind operations; device access.
    - Package management and service control (`systemctl
      start/stop/restart`, `apt`, …) — diagnostics are read-only (T07).
    - Reading credential material: secret-file denial reuses
      `DesktopPathSecurity` classification (credential/secret/private-key
      files denied even inside allowed roots — e.g. `.env`, `*.pem`,
      `id_rsa*`).
    - Killing processes not started by the same run (T06); signalling the
      daemon itself or the SlavikAI server.
    - GUI control (`desktop.control`) — absent in v1; observation only if
      the owner enables it under the T10 gate.
    - Network listeners opened by the daemon; **spawned processes have
      no network access in v1** (see §10a).
    - Persistence: writing systemd units, cron entries, autostart files, or
      shell rc files (outside allowed roots by construction; named here so
      a future root change cannot silently enable it).

10a. **Exec model (owner decision 2026-11-23).** `host.process.run`
    executes only **approved command profiles**. Each profile fixes: the
    absolute binary path, the allowed argument patterns, the working
    directory (inside allowed roots), and a minimal environment. An
    `argv[0]` allowlist alone is explicitly insufficient. Spawned
    processes run **without network** in v1 and under OS-level
    file-access restrictions; if the execution environment cannot
    enforce those restrictions, `host.process.run` is unavailable
    (fail-closed — the capability is not offered degraded). Runtime cap
    is 120 s. stdout/stderr are bounded **during collection** (streaming
    cap), not truncated after the fact. On STOP or timeout the daemon
    terminates the **entire process tree** of the run.

11. **Delete semantics (owner decision 2026-11-23).** `host.fs.delete`
    moves to trash only (existing `DesktopFileDeleteTool` already has a
    `trash_root`); hard delete is not exposed in v1.

12. **Adoption rule (delete-vs-wrap).** The bridge FS API reuses
    `DesktopPathSecurity` internals (E03-T04 note). Once `owner_host` is
    adopted, the direct in-process desktop filesystem/shell tools are
    **removed**, not kept as a parallel path — one host boundary, one
    implementation (project anti-duplicate-path rule). The sandbox lane
    (`shared/sandbox.py`, container backend) is unaffected: it remains the
    target for untrusted-code execution, the bridge is the target for
    owner-host work.

## Alternatives considered

- **Keep in-process desktop tools, add more checks:** rejected; leaves host
  authority ambient in the server process, blocks server sandboxing, and
  gives no transport identity or independent hardening boundary.
- **TCP loopback daemon with a bearer token:** rejected; a token is a
  copyable secret usable by any local process that reads it, while
  `SO_PEERCRED` binds identity to the connecting process; loopback TCP also
  invites accidental exposure via forwarding.
- **Root daemon with capability dropping per request:** rejected for v1;
  the ticket's owner gate reserves root for a separate decision, and no
  v1 capability needs privileges beyond the owner's UID.
- **Bridge as second policy engine (bridge decides ask/deny):** rejected;
  duplicates DesktopPolicyRuntime, splits the audit/policy story, and
  contradicts the E02 scoped-authorization decision (one policy engine,
  grants in ToolGateway/RunPolicy).
- **AF_UNIX + SO_PEERCRED, non-root, hard bounds daemon-side, policy in
  ToolGateway:** selected.

## Consequences and boundaries

- Owner decisions are recorded (2026-11-23); E03-T02 may start. T02
  implements lifecycle/auth only (`ping/version/capabilities`).
- The server gains a new failure mode (daemon down → `owner_host` target
  unavailable, typed error, no silent fallback to in-process desktop tools —
  fail-closed per project rules).
- Desktop tools migration is a later, separately reviewed step (T04
  delete-vs-wrap); until adoption completes, both paths exist and the ADR
  does not claim the in-process path is already removed.
- Same-UID malware remains out of scope for transport auth (attacker class
  5); the bridge does not make a compromised owner account safe.
- Exact wire schema, frame limits, and timeout values are T03 decisions
  within this model; root list and forbidden-operations list changes after
  v1 require an ADR amendment, not a config tweak.

## Owner decisions (2026-11-23)

All nine forks from the proposed draft are resolved:

1. **Allowed roots** — default only `~/.local/share/slavikai/host-workspace`
   (RW); owner explicitly allows specific subdirectories of `~/Project`,
   `~/Documents`, `~/Downloads` with per-subdirectory mode; SlavikAI
   checkouts, bridge launch files, policy, and runtime state are
   write-excluded even inside a root; secret denial applies over the
   allowlist; `/etc` is a hard bridge deny (stricter than the in-process
   `ask` classification).
2. **Exec** — approved command profiles only (binary path, argument
   patterns, working directory, minimal environment); `argv[0]`
   allowlist alone is insufficient; spawned processes have no network
   in v1 and run under OS-level file restrictions, else
   `host.process.run` is unavailable; 120 s cap; stdout/stderr bounded
   during collection; STOP/timeout kills the whole process tree.
3. **Lanes** — owner-role runs only in v1.
4. **GUI** — none in v1; separate owner decision in T10.
5. **Socket** — `$XDG_RUNTIME_DIR`, closed permissions, `SO_PEERCRED`;
   run authorization confirmed by the trusted runtime independently of
   `principal`/`run_id` in request JSON.
6. **Delete** — trash-only.
7. **Secret-file denial inside roots** — yes, by `DesktopPathSecurity`
   classification.
8. **Process limits** — 120 s cap and bounded output as in §10a.
9. **Daemon lifecycle** — systemd user service, non-root.
