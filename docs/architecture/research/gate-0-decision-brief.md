# Gate 0 decision brief

**Status:** preserved decision brief, 2026-09-24. The complete evidence and unresolved
details remain in [gate-0-closure-matrix.md](gate-0-closure-matrix.md),
[capability-boundary-proposal.md](capability-boundary-proposal.md) and
[open-questions.md](open-questions.md). The accepted system-level map is in
`system/`; this brief does not accept unanswered detailed choices or claim runtime.

## Product choices already made

| Decision | Accepted outcome | Remaining design |
| --- | --- | --- |
| ADR-0001 | Subscription-backed OpenAI/ChatGPT route is mandatory and distinct from API billing. | Compliant entitlement/auth/transport and qualification. |
| ADR-0002 | Terminal cause and result disposition are separate; failed/cancelled/aborted final leads with cause, then accepted partial work. Only user or preapproved criteria accept an incomplete result. | Reconciled Lifecycle contract. |
| ADR-0003 | New work may start on schedule/external event without a new message only under rules preapproved by the user. | Detailed rule/firing protocol and task creation. |
| ADR-0004 | Work Initiation owns approved trigger rules and firing history; Lifecycle validates/accepts task creation. | Exact deduplication, overlap, catch-up and cross-owner recovery. |
| ADR-0005 | Local Inference Operations owns the host engine lifecycle separately from Model Access route qualification/selection. | Acquisition, tuning, authorization, resource and recovery contracts. |
| ADR-0006 | Resource Governance owns shared allocation; each domain owns material history; System Evaluation owns cross-task quality findings. | Accounting, event/replay and eval methodology. |
| ADR-0007 | Owner provider keys are owner-only by default; another principal needs an explicit delegation rule for use. | Exact credential handle, route checks, rotation/revocation and runtime rollout. |
| ADR-0008 | Separate request-scoped media operations are sufficient; no continuous voice/visual session owner is required. | Per-request consent, provenance, egress and artifact handoffs. |
| ADR-0009 | External tool providers/extensions are Target scope, with permission and version governance separate from skills. | Provider identity, version/revocation, credentials and call-time policy contract. |
| ADR-0010 | Superseded accepted revision receives status and lineage to its replacement, without a separate final solely for supersession. | Reconcile Lifecycle state model and publication rules with ADR-0002. |

These product choices are accepted in their ADRs. Detailed protocol, storage
and PR breakdown belong to the dependent sections.

## Engineering boundary left after the choices

ADR-0003 settled the **presence** of autonomous initiation, and ADR-0004
settled its owner boundary. OQ-CD-04 still needs exact rule/firing behavior,
including principal/source binding, deduplication, missed events and overlap.
ADR-0005 settles the local-engine versus model-route owner split; OQ-MA-12/13
retain product and operational scope details.
Task/run control must validate the resulting proposal;
background execution resumes existing work and communication delivers its
status/final. The same owner split is applied to the other selected capabilities
in the four accepted coarse `system/` documents.
Their detailed contracts remain dependent work.

**Next step:** reconcile the Lifecycle snapshot with ADR-0002/0010 before
section 4 contract integration or runtime work.
