# Generic OpenAI-compatible provider onboarding

Owner Settings accepts a display name, Base URL and API key. A probe calls the normalized
`/models` endpoint with the supplied key. HTTP 401 rejects the key. Saving creates an endpoint
instance ID without a model, stores its name and Base URL in `.run/ui_settings.json`, and stores
the key in the existing mode-0600 `config/api_keys.json` boundary. Saving does not select a model.
Remote endpoints require HTTPS; HTTP is accepted only for loopback runtimes.

The chat session model picker discovers models through `/models` when the owner opens an instance.
If the catalog succeeds, the selected ID must be in it. If the catalog is unavailable or
unsupported, the picker permits a manual opaque model ID; HTTP 401 never permits this fallback.
The selected ID belongs to session state. The existing OpenAI-compatible `LocalHttpBrain` sends
text/chat completions to the stored Base URL. Dynamic instances are fail-closed for native tools;
Auto reports `native_tools_required` and no tool schema is sent to an unqualified endpoint.
Capability qualification is a future increment. Built-in `local` capability and the public
`/v1` proxy model `slavik` are unchanged.

Dynamic instances use owner credentials and are server-side default-deny for members at model
listing, discovery, selection and completion boundaries until explicit delegation is implemented.

TODO: Add first-class `Sign in with ChatGPT` subscription access through a separately designed
auth/entitlement route as required by ADR-0001. An API key provider instance is never treated
as subscription access. Owner credential delegation remains the separate ADR-0007 target gap.
