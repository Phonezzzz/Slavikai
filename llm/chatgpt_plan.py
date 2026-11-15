"""Подписка ChatGPT: OAuth-регистрации и токены, привязанные к principal."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, field, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socket import socket
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import uuid4

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from shared.models import JSONValue

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
CALLBACK_PATH = "/auth/callback"
SCOPES = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "resource.invoke",
    "chatgpt.tokens.use.direct",
)
PLAN_SCOPES = frozenset(SCOPES[3:])
PROVIDER_PREFIX = "chatgpt-plan-"
DYNAMIC_CLIENT = "dynamic_agent_client"
ATTEMPT_LIFETIME = 300
MAX_RECORD = 4 * 1024 * 1024
INVALID_REFRESH = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)


class ChatGPTPlanError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 502,
        request_id: str | None = None,
        body_shape: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.request_id = request_id
        self.body_shape = body_shape
        self.upstream_status: int | None = None


def _storage_error() -> ChatGPTPlanError:
    return ChatGPTPlanError(
        "chatgpt_plan_storage", "Хранилище аккаунтов ChatGPT plan недоступно.", status=503
    )


def _account_error() -> ChatGPTPlanError:
    return ChatGPTPlanError(
        "chatgpt_plan_disconnected",
        "Выбранный доступ ChatGPT plan отсутствует или изменился.",
        status=409,
    )


def _callback_error() -> ChatGPTPlanError:
    return ChatGPTPlanError(
        "chatgpt_plan_callback", "Не удалось подтвердить запрос авторизации.", status=400
    )


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected non-empty string")
    return value


def _b64(value: str) -> bytes:
    return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)


def _verify_identity(
    token: str, jwks: dict[str, object], client: str, nonce: str
) -> dict[str, object]:
    try:
        header, payload, signature = token.split(".")
        fields = _object(json.loads(_b64(header)))
        claims = _object(json.loads(_b64(payload)))
        kid = _string(fields, "kid")
        keys = jwks.get("keys")
        if fields.get("alg") != "RS256" or "crit" in fields or not isinstance(keys, list):
            raise ValueError("unexpected signing algorithm")
        matches = [key for key in keys if isinstance(key, dict) and key.get("kid") == kid]
        if len(matches) != 1:
            raise ValueError("unknown signing key")
        key = matches[0]
        if key.get("kty") != "RSA" or key.get("use", "sig") != "sig":
            raise ValueError("invalid key")
        if key.get("alg", "RS256") != "RS256":
            raise ValueError("unexpected key algorithm")
        exponent = int.from_bytes(_b64(_string(key, "e")), "big")
        modulus = int.from_bytes(_b64(_string(key, "n")), "big")
        rsa.RSAPublicNumbers(exponent, modulus).public_key().verify(
            _b64(signature),
            f"{header}.{payload}".encode("ascii"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        now = time.time()
        audience = claims.get("aud")
        expiration = claims.get("exp")
        issued = claims.get("iat")
        not_before = claims.get("nbf", 0)
        if (
            claims.get("iss") != ISSUER
            or (audience != client and audience != [client])
            or claims.get("azp", client) != client
            or claims.get("nonce") != nonce
            or type(expiration) is not int
            or expiration <= now
            or type(issued) is not int
            or issued > now + 5
            or type(not_before) is not int
            or not_before > now + 5
        ):
            raise ValueError("invalid token claims")
        _string(claims, "sub")
        return claims
    except (ValueError, InvalidSignature):
        raise ChatGPTPlanError(
            "chatgpt_plan_identity", "Идентичность аккаунта ChatGPT не подтверждена.", status=400
        ) from None


def _upstream_error(status: int, code: object, *, refreshing: bool) -> ChatGPTPlanError:
    if refreshing and isinstance(code, str) and code in INVALID_REFRESH:
        return ChatGPTPlanError(
            "chatgpt_plan_revoked",
            "Сессия ChatGPT отозвана или истекла. Войдите снова.",
            status=409,
        )
    if refreshing and code == "invalid_client":
        return ChatGPTPlanError(
            "chatgpt_plan_registration", "OpenAI отклонил регистрацию клиента ChatGPT plan."
        )
    if code == "subscription_sharing_usage_limit_exceeded":
        return ChatGPTPlanError(
            "chatgpt_plan_quota",
            "Лимит ChatGPT plan исчерпан. Проверьте Usage в ChatGPT.",
            status=429,
        )
    if status == 429:
        return ChatGPTPlanError(
            "chatgpt_plan_rate_limit", "Превышена частота запросов ChatGPT plan.", status=429
        )
    if status == 401 or code == "subscription_sharing_invalid_user":
        return ChatGPTPlanError(
            "chatgpt_plan_authorization", "OpenAI отклонил авторизацию ChatGPT plan.", status=401
        )
    if status == 403 or code == "subscription_sharing_user_not_eligible":
        return ChatGPTPlanError(
            "chatgpt_plan_blocked", "OpenAI ограничил доступ к ChatGPT plan.", status=403
        )
    if status >= 500:
        return ChatGPTPlanError(
            "chatgpt_plan_upstream_unavailable", "OpenAI временно недоступен.", status=503
        )
    return ChatGPTPlanError("chatgpt_plan_upstream", "OpenAI отклонил запрос ChatGPT plan.")


def _request(
    method: str,
    url: str,
    *,
    data: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    empty: bool = False,
) -> dict[str, object]:
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.request(
                method, url, timeout=(4, 8), allow_redirects=False, data=data, headers=headers
            )
            if response.status_code == 200:
                return {} if empty else _object(response.json())
            body_shape = "unknown"
            try:
                body = _object(response.json())
                error = body.get("error")
                if isinstance(error, dict):
                    body_shape = "structured"
                    code = error.get("code")
                elif isinstance(error, str):
                    body_shape = "oauth"
                    code = error
                else:
                    body_shape = "detail" if isinstance(body.get("detail"), str) else "unknown"
                    code = None
            except ValueError:
                code = None
            failure = _upstream_error(
                response.status_code,
                code,
                refreshing=(url == TOKEN_URL and (data or {}).get("grant_type") == "refresh_token"),
            )
            request_id = response.headers.get("x-request-id", "")
            failure.request_id = (
                request_id if re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", request_id) else None
            )
            failure.upstream_status = response.status_code
            failure.body_shape = body_shape
            raise failure
    except requests.ConnectionError:
        raise ChatGPTPlanError(
            "chatgpt_plan_transport", "Не удалось связаться с OpenAI.", status=503
        ) from None
    except requests.Timeout:
        raise ChatGPTPlanError(
            "chatgpt_plan_transport", "Превышено время ожидания ответа OpenAI.", status=503
        ) from None
    except (requests.RequestException, ValueError):
        raise ChatGPTPlanError(
            "chatgpt_plan_response", "Не удалось прочитать ответ OpenAI."
        ) from None


@dataclass(frozen=True)
class Account:
    id: str
    client_id: str
    subject: str
    label: str
    status: str = "disconnected"
    provider: str | None = None
    access_token: str = field(default="", repr=False)
    refresh_token: str = field(default="", repr=False)
    id_token: str = field(default="", repr=False)
    expires: int = 0
    scopes: tuple[str, ...] = ()

    def clear(self, status: str) -> Account:
        return replace(
            self,
            status=status,
            provider=None,
            access_token="",
            refresh_token="",
            id_token="",
            expires=0,
            scopes=(),
        )


@dataclass
class Registrations:
    accounts: dict[str, Account] = field(default_factory=dict)
    attempt: str | None = None
    selected: str | None = None
    expires: int = 0
    status: str | None = None

    def finished(self, status: str) -> None:
        self.attempt = None
        self.selected = None
        self.expires = 0
        self.status = status


def _checked_file(fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("invalid credentials file")


@dataclass(frozen=True)
class _Storage:
    directory_fd: int
    name: str

    def read(self) -> dict[str, object]:
        try:
            fd = os.open(
                self.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.directory_fd
            )
        except FileNotFoundError:
            return {}
        with os.fdopen(fd, encoding="utf-8") as handle:
            _checked_file(handle.fileno())
            raw = handle.read(MAX_RECORD + 1)
        if len(raw) > MAX_RECORD:
            raise ValueError("credentials record too large")
        return _object(json.loads(raw))

    def save(self, record: dict[str, object]) -> None:
        raw = json.dumps(record, separators=(",", ":"))
        if len(raw) > MAX_RECORD:
            raise ValueError("credentials record too large")
        temporary = "." + uuid4().hex
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.directory_fd,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary, self.name, src_dir_fd=self.directory_fd, dst_dir_fd=self.directory_fd
            )
            os.fsync(self.directory_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.directory_fd)
            except FileNotFoundError:
                pass

    def registrations(self) -> Registrations:
        raw = self.read()
        accounts = {}
        for account_id, value in _object(raw.get("accounts", {})).items():
            data = _object(value)
            if account_id != data.get("id"):
                raise ValueError("invalid account ID")
            account = Account(**data)  # type: ignore[arg-type]
            if (
                any(
                    not isinstance(val, str)
                    for val in (
                        account.id,
                        account.client_id,
                        account.subject,
                        account.label,
                        account.status,
                        account.access_token,
                        account.refresh_token,
                        account.id_token,
                    )
                )
                or (account.provider is not None and not isinstance(account.provider, str))
                or type(account.expires) is not int
                or not isinstance(account.scopes, (list, tuple))
                or any(not isinstance(scope, str) for scope in account.scopes)
            ):
                raise ValueError("invalid account record")
            accounts[account_id] = replace(account, scopes=tuple(account.scopes))
        expires = raw.get("expires", 0)
        attempt, selected, status = (raw.get(key) for key in ("attempt", "selected", "status"))
        if (
            (attempt is not None and not isinstance(attempt, str))
            or (selected is not None and not isinstance(selected, str))
            or (status is not None and not isinstance(status, str))
            or type(expires) is not int
        ):
            raise ValueError("invalid authorization record")
        return Registrations(accounts, attempt, selected, expires, status)


class CallbackServer(HTTPServer):
    def __init__(self, service: ChatGPTPlan, attempt: _Authorization) -> None:
        self.service = service
        self.attempt = attempt
        super().__init__(("127.0.0.1", 0), CallbackHandler)
        self.timeout = 0.25
        self.host = f"127.0.0.1:{self.server_port}"

    def get_request(self) -> tuple[socket, tuple[str, int]]:
        connection, address = super().get_request()
        connection.settimeout(2)
        return connection, address

    def handle_error(self, request: object, client_address: object) -> None:
        # BaseServer выбрасывает в stderr traceback с HTTP-запросом.
        pass


class CallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, CallbackServer)
        try:
            # BaseHTTPRequestHandler сворачивает начальные двойные слеши в self.path.
            path, separator, query = self.requestline.split()[1].partition("?")
            if (
                self.headers.get_all("Host") != [server.host]
                or path != CALLBACK_PATH
                or not separator
                or len(query) > 16384
            ):
                raise _callback_error()
            params = parse_qs(query, keep_blank_values=True, max_num_fields=12, errors="strict")
            if any(len(values) != 1 for values in params.values()):
                raise _callback_error()
            result = server.service._complete(
                server, {key: values[0] for key, values in params.items()}
            )
            if result == "disabled":
                message = "Аккаунт сохранён. Доступ к ChatGPT plan не предоставлен."
            else:
                message = "Аккаунт ChatGPT подключён. Вернитесь в SlavikAI."
            status = 200
        except (ChatGPTPlanError, ValueError):
            message = "Авторизация не завершена. Вернитесь в SlavikAI и проверьте её статус."
            status = 400
        body = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)


@dataclass(frozen=True)
class _Authorization:
    id: str
    principal: str
    state: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    verifier: str = field(default_factory=lambda: secrets.token_urlsafe(48), repr=False)
    expires: float = field(default_factory=lambda: time.monotonic() + ATTEMPT_LIFETIME)
    stop: threading.Event = field(default_factory=threading.Event, repr=False)


def _with_tokens(account: Account, data: dict[str, object], scopes: tuple[str, ...]) -> Account:
    try:
        duration = data.get("expires_in")
        if _string(data, "token_type").lower() != "bearer" or (
            type(duration) is not int or not 0 < duration <= 86400
        ):
            raise ValueError("invalid token response")
        access_token = _string(data, "access_token")
        refresh_token = _string(data, "refresh_token")
        if any(any(char.isspace() for char in token) for token in (access_token, refresh_token)):
            raise ValueError("invalid token")
        return replace(
            account,
            status="connected",
            access_token=access_token,
            refresh_token=refresh_token,
            expires=int(time.time()) + duration,
            scopes=scopes,
        )
    except ValueError:
        raise ChatGPTPlanError(
            "chatgpt_plan_token_response", "OpenAI вернул неполный набор токенов."
        ) from None


def _disabled(account: Account, data: dict[str, object], scopes: tuple[str, ...]) -> Account:
    refresh = data.get("refresh_token")
    return replace(
        account,
        status="disabled",
        provider=None,
        access_token="",
        expires=0,
        scopes=scopes,
        refresh_token=refresh if isinstance(refresh, str) else "",
    )


class ChatGPTPlan:
    """Локальная регистрация аккаунтов. Блокировка охватывает ротацию refresh token."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        self._listeners: dict[str, CallbackServer] = {}
        self._closed = threading.Event()

    @contextmanager
    def _store(self, name: str) -> Iterator[_Storage]:
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                directory = os.fstat(directory_fd)
                if directory.st_uid != os.getuid() or directory.st_mode & 0o077:
                    raise ValueError("invalid credential directory")
                lock = os.open(
                    f"{name}.lock",
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    _checked_file(lock)
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    yield _Storage(directory_fd, name)
                finally:
                    os.close(lock)
            finally:
                os.close(directory_fd)
        except (OSError, TypeError, ValueError):
            raise _storage_error() from None

    def _principal_store(self, principal: str) -> AbstractContextManager[_Storage]:
        name = hashlib.sha256(principal.encode()).hexdigest() + ".json"
        return self._store(name)

    def _host(self) -> str:
        with self._store("host.json") as storage:
            record = storage.read()
            if record:
                return _string(record, "id")
            host_id = f"urn:uuid:{uuid4()}"
            storage.save({"id": host_id})
            return host_id

    def begin(
        self, principal: str, account_id: str | None = None, *, reconsent: bool = False
    ) -> str:
        host = self._host()
        attempt = _Authorization(uuid4().hex, principal)
        try:
            server = CallbackServer(self, attempt)
        except OSError:
            raise ChatGPTPlanError(
                "chatgpt_plan_loopback", "Локальный callback ChatGPT plan недоступен.", status=503
            ) from None

        try:
            with self._lock:
                if self._closed.is_set():
                    raise _callback_error()
                with self._principal_store(principal) as storage:
                    records = storage.registrations()
                    selected = records.accounts.get(account_id) if account_id is not None else None
                    if account_id is not None and selected is None:
                        raise _account_error()
                    records.attempt = attempt.id
                    records.selected = account_id
                    records.expires = int(time.time()) + ATTEMPT_LIFETIME
                    records.status = "pending"
                    storage.save(asdict(records))
                old = self._listeners.get(principal)
                if old is not None:
                    old.attempt.stop.set()
                self._listeners[principal] = server
                threading.Thread(target=self._listen, args=(server,), daemon=True).start()
        except Exception:
            server.server_close()
            raise

        digest = hashlib.sha256(attempt.verifier.encode("ascii")).digest()
        query = {
            "client_id": selected.client_id if selected else DYNAMIC_CLIENT,
            "ext_agent_host_id": host,
            "response_type": "code",
            "redirect_uri": self._redirect(server),
            "scope": " ".join(SCOPES),
            "resource": RESOURCE,
            "state": attempt.state,
            "nonce": attempt.nonce,
            "code_challenge_method": "S256",
            "code_challenge": base64.urlsafe_b64encode(digest).decode().rstrip("="),
        }
        if selected is None:
            query["agent_name_hint"] = "SlavikAI"
        if reconsent:
            query["prompt"] = "consent"
        return f"{ISSUER}/api/accounts/authorize?{urlencode(query)}"

    @staticmethod
    def _redirect(server: CallbackServer) -> str:
        return f"http://{server.host}{CALLBACK_PATH}"

    def _listen(self, server: CallbackServer) -> None:
        attempt = server.attempt
        try:
            while not attempt.stop.is_set() and time.monotonic() < attempt.expires:
                server.handle_request()
        except (OSError, ValueError):
            if not attempt.stop.is_set():
                raise
        finally:
            server.server_close()
            with self._lock:
                if self._listeners.get(attempt.principal) is server:
                    del self._listeners[attempt.principal]

    def _complete(self, server: CallbackServer, query: dict[str, str]) -> str:
        attempt = server.attempt
        if (
            attempt.stop.is_set()
            or time.monotonic() >= attempt.expires
            or not secrets.compare_digest(query.get("state", "").encode(), attempt.state.encode())
        ):
            raise _callback_error()

        with self._principal_store(attempt.principal) as storage:
            records = storage.registrations()
            if (
                records.attempt != attempt.id
                or attempt.stop.is_set()
                or time.monotonic() >= attempt.expires
            ):
                raise _callback_error()
            attempt.stop.set()
            try:
                account = self._exchange(server, query, records)
                if self._closed.is_set():
                    raise _callback_error()
            except ChatGPTPlanError:
                records.finished("failed")
                storage.save(asdict(records))
                raise
            records.accounts[account.id] = account
            records.finished(account.status)
            storage.save(asdict(records))
            return account.status

    def _exchange(
        self, server: CallbackServer, query: dict[str, str], records: Registrations
    ) -> Account:
        if "error" in query:
            raise _callback_error()
        selected = records.accounts.get(records.selected) if records.selected else None
        client = query.get("client_id", selected.client_id if selected else "")
        code = query.get("code")
        if (
            not code
            or not client
            or client == DYNAMIC_CLIENT
            or (selected is not None and client != selected.client_id)
        ):
            raise _callback_error()

        data = _request(
            "POST",
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": client,
                "code": code,
                "code_verifier": server.attempt.verifier,
                "redirect_uri": self._redirect(server),
                "resource": RESOURCE,
            },
        )
        try:
            token = _string(data, "id_token")
        except ValueError:
            raise _callback_error() from None
        claims = _verify_identity(token, _request("GET", JWKS_URL), client, server.attempt.nonce)
        subject = _string(claims, "sub")
        existing = next(
            (item for item in records.accounts.values() if item.client_id == client), None
        )
        if (existing is not None and existing.subject != subject) or (
            selected is not None and selected.subject != subject
        ):
            raise _callback_error()

        key = existing.id if existing else uuid4().hex
        email = claims.get("email")
        label = (
            email[:254]
            if isinstance(email, str) and claims.get("email_verified") is True
            else ("ChatGPT")
        )
        label = existing.label if existing else f"{label} ({key[:8]})"
        account = Account(key, client, subject, label, id_token=token)
        scope = data.get("scope")
        scopes = tuple(scope.split()) if isinstance(scope, str) else ()
        if not PLAN_SCOPES.issubset(scopes):
            return _disabled(account, data, scopes)
        connected = _with_tokens(
            replace(account, provider=PROVIDER_PREFIX + uuid4().hex), data, scopes
        )
        if "at_hash" in claims:
            try:
                half_digest = hashlib.sha256(connected.access_token.encode("ascii")).digest()[:16]
                expected = base64.urlsafe_b64encode(half_digest).decode().rstrip("=")
                paired = claims.get("at_hash")
                if not isinstance(paired, str) or not secrets.compare_digest(
                    paired.encode(), expected.encode()
                ):
                    raise _callback_error()
            except UnicodeError:
                raise _callback_error() from None
        return connected

    def authorization_status(self, principal: str) -> str | None:
        with self._principal_store(principal) as storage:
            records = storage.registrations()
            if records.status == "pending" and records.expires <= time.time():
                return "expired"
            return records.status

    def accounts(self, principal: str) -> list[dict[str, JSONValue]]:
        with self._principal_store(principal) as storage:
            return [
                {
                    "id": account.id,
                    "label": account.label,
                    "provider": account.provider,
                    "status": account.status,
                }
                for account in storage.registrations().accounts.values()
            ]

    def validate_grant(self, principal: str, provider: str) -> None:
        with self._principal_store(principal) as storage:
            self._account_for_grant(storage.registrations(), provider)

    def access(self, principal: str, provider: str) -> str:
        with self._principal_store(principal) as storage:
            records = storage.registrations()
            return self._access(storage, records, provider)

    @staticmethod
    def _account_for_grant(records: Registrations, provider: str) -> Account:
        account = next(
            (item for item in records.accounts.values() if item.provider == provider), None
        )
        if (
            account is None
            or not provider.startswith(PROVIDER_PREFIX)
            or account.status != "connected"
            or not account.access_token
            or not account.refresh_token
            or not PLAN_SCOPES.issubset(account.scopes)
        ):
            raise _account_error()
        return account

    def _access(self, storage: _Storage, records: Registrations, provider: str) -> str:
        account = self._account_for_grant(records, provider)
        if account.expires > time.time() + 20:
            return account.access_token
        try:
            data = _request(
                "POST",
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": account.client_id,
                    "refresh_token": account.refresh_token,
                    "resource": RESOURCE,
                },
            )
        except ChatGPTPlanError as error:
            if error.code == "chatgpt_plan_revoked":
                records.accounts[account.id] = account.clear("expired")
                storage.save(asdict(records))
            raise

        scope = data.get("scope")
        scopes = (
            account.scopes
            if "scope" not in data
            else tuple(scope.split())
            if isinstance(scope, str)
            else ()
        )
        if not PLAN_SCOPES.issubset(scopes):
            records.accounts[account.id] = _disabled(account, data, scopes)
            storage.save(asdict(records))
            raise ChatGPTPlanError(
                "chatgpt_plan_permission", "OpenAI отозвал разрешение ChatGPT plan.", status=403
            )
        updated = _with_tokens(account, data, scopes)
        records.accounts[account.id] = updated
        storage.save(asdict(records))
        return updated.access_token

    def catalog(self, principal: str, provider: str) -> dict[str, str]:
        with self._principal_store(principal) as storage:
            access = self._access(storage, storage.registrations(), provider)
            data = _request(
                "GET", f"{RESOURCE}/models", headers={"Authorization": f"Bearer {access}"}
            )
        models = data.get("models")
        if not isinstance(models, list):
            raise ChatGPTPlanError(
                "chatgpt_plan_catalog", "Каталог моделей ChatGPT plan недоступен."
            )
        return {
            item["slug"]: item["display_name"]
            for item in models
            if isinstance(item, dict)
            and item.get("visibility") == "list"
            and isinstance(item.get("slug"), str)
            and item["slug"]
            and isinstance(item.get("display_name"), str)
            and item["display_name"]
        }

    def disconnect(self, principal: str, account_id: str) -> bool:
        with self._principal_store(principal) as storage:
            records = storage.registrations()
            account = records.accounts.get(account_id)
            if account is None:
                raise _account_error()
            if records.selected == account_id:
                records.finished("failed")
            records.accounts[account_id] = account.clear("disconnected")
            storage.save(asdict(records))
            if not account.refresh_token:
                return False
            return self._revoke(account)

    @staticmethod
    def _revoke(account: Account) -> bool:
        endpoint: str | None = None
        for delay in (0, 0.2, 0.5):
            if delay:
                time.sleep(delay)
            try:
                if endpoint is None:
                    discovered = _string(_request("GET", DISCOVERY_URL), "revocation_endpoint")
                    parsed = urlsplit(discovered)
                    if (
                        parsed.scheme != "https"
                        or parsed.netloc != "auth.openai.com"
                        or not parsed.path.startswith("/")
                        or parsed.fragment
                        or parsed.query
                    ):
                        return False
                    endpoint = discovered
                _request(
                    "POST",
                    endpoint,
                    empty=True,
                    data={
                        "token": account.refresh_token,
                        "token_type_hint": "refresh_token",
                        "client_id": account.client_id,
                    },
                )
                return True
            except ValueError:
                return False
            except ChatGPTPlanError as error:
                if error.status != 503:
                    return False
        return False

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            for server in self._listeners.values():
                server.attempt.stop.set()
                server.server_close()
