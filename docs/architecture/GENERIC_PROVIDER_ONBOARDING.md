# Generic OpenAI-compatible provider onboarding

Owner Settings accepts a display name, Base URL and API key. A probe calls the normalized
`/models` endpoint with the supplied key. HTTP 401 rejects the key; an unavailable or
unsupported `/models` endpoint is reported and permits a manual opaque model ID. Saving creates
an instance ID, stores non-secret metadata in `.run/ui_settings.json`, and stores the key in the
existing mode-0600 `config/api_keys.json` boundary. Saving does not select the model.

The existing UI model picker selects the instance and model explicitly. The existing
OpenAI-compatible `LocalHttpBrain` sends chat completions to the stored Base URL; native tool
calls continue through the existing agent loop and `ToolGateway`. Built-in provider selection
and public `/v1` proxy model `slavik` are unchanged.

TODO: Add first-class `Sign in with ChatGPT` subscription access through a separately designed
auth/entitlement route as required by ADR-0001. An API key provider instance is never treated
as subscription access. Owner credential delegation remains the separate ADR-0007 target gap.
