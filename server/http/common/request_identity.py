from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import aiohttp
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

IdentityRole = Literal["owner", "member", "automation"]
IdentityMethod = Literal["cloudflare_access", "bearer", "cookie", "local"]


@dataclass(frozen=True, slots=True)
class RequestIdentity:
    principal_id: str
    role: IdentityRole
    auth_method: IdentityMethod
    email: str | None = None


@dataclass(frozen=True, slots=True)
class VerifiedAccessClaims:
    email: str
    subject: str | None = None


class CloudflareAccessTokenError(ValueError):
    pass


class CloudflareAccessVerifier(Protocol):
    async def verify(self, token: str) -> VerifiedAccessClaims: ...


JWKSFetcher = Callable[[], Awaitable[object]]


class CloudflareAccessJWTVerifier:
    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_fetcher: JWKSFetcher | None = None,
        now: Callable[[], float] = time.time,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        self.issuer = issuer.rstrip("/")
        self.audience = audience
        self.jwks_url = f"{self.issuer}/cdn-cgi/access/certs"
        self._jwks_fetcher = jwks_fetcher
        self._now = now
        self._cache_ttl_seconds = max(1.0, cache_ttl_seconds)
        self._cached_keys: dict[str, Mapping[str, object]] = {}
        self._cache_expires_at = 0.0
        self._cache_lock = asyncio.Lock()

    async def verify(self, token: str) -> VerifiedAccessClaims:
        header_segment, payload_segment, signature_segment = _split_jwt(token)
        header = _decode_json_segment(header_segment, "header")
        payload = _decode_json_segment(payload_segment, "payload")
        if header.get("alg") != "RS256":
            raise CloudflareAccessTokenError("Cloudflare Access JWT must use RS256")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid.strip():
            raise CloudflareAccessTokenError("Cloudflare Access JWT kid is missing")
        jwk = await self._key_for_kid(kid.strip())
        public_key = _rsa_public_key(jwk)
        try:
            public_key.verify(
                _decode_base64url(signature_segment),
                f"{header_segment}.{payload_segment}".encode("ascii"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
        except Exception as exc:  # noqa: BLE001
            raise CloudflareAccessTokenError("Cloudflare Access JWT signature is invalid") from exc
        self._validate_registered_claims(payload)
        email = normalize_email(payload.get("email"))
        if email is None:
            raise CloudflareAccessTokenError("Cloudflare Access JWT email is missing")
        subject_raw = payload.get("sub")
        subject = (
            subject_raw.strip() if isinstance(subject_raw, str) and subject_raw.strip() else None
        )
        return VerifiedAccessClaims(email=email, subject=subject)

    async def _key_for_kid(self, kid: str) -> Mapping[str, object]:
        now = self._now()
        cached = self._cached_keys.get(kid)
        if cached is not None and now < self._cache_expires_at:
            return cached
        async with self._cache_lock:
            now = self._now()
            cached = self._cached_keys.get(kid)
            if cached is not None and now < self._cache_expires_at:
                return cached
            keys = await self._fetch_keys()
            self._cached_keys = keys
            self._cache_expires_at = now + self._cache_ttl_seconds
            selected = keys.get(kid)
            if selected is None:
                raise CloudflareAccessTokenError("Cloudflare Access JWT kid is unknown")
            return selected

    async def _fetch_keys(self) -> dict[str, Mapping[str, object]]:
        if self._jwks_fetcher is not None:
            raw = await self._jwks_fetcher()
        else:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.jwks_url) as response:
                    if response.status != 200:
                        raise CloudflareAccessTokenError(
                            f"Cloudflare Access JWKS request failed: HTTP {response.status}"
                        )
                    raw = await response.json()
        if not isinstance(raw, dict) or not isinstance(raw.get("keys"), list):
            raise CloudflareAccessTokenError("Cloudflare Access JWKS payload is invalid")
        keys: dict[str, Mapping[str, object]] = {}
        for item in raw["keys"]:
            if not isinstance(item, dict):
                continue
            kid = item.get("kid")
            if isinstance(kid, str) and kid.strip():
                keys[kid.strip()] = item
        if not keys:
            raise CloudflareAccessTokenError("Cloudflare Access JWKS contains no keys")
        return keys

    def _validate_registered_claims(self, payload: Mapping[str, object]) -> None:
        if payload.get("iss") != self.issuer:
            raise CloudflareAccessTokenError("Cloudflare Access JWT issuer is invalid")
        audience = payload.get("aud")
        audiences = [audience] if isinstance(audience, str) else audience
        if not isinstance(audiences, list) or self.audience not in audiences:
            raise CloudflareAccessTokenError("Cloudflare Access JWT audience is invalid")
        now = self._now()
        exp = _numeric_date(payload.get("exp"), "exp")
        nbf = _numeric_date(payload.get("nbf"), "nbf")
        if now >= exp:
            raise CloudflareAccessTokenError("Cloudflare Access JWT is expired")
        if now < nbf:
            raise CloudflareAccessTokenError("Cloudflare Access JWT is not active yet")


def normalize_email(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if not normalized or "@" not in normalized:
        return None
    local, domain = normalized.rsplit("@", 1)
    if not local or not domain:
        return None
    return normalized


def principal_id_for_email(email: str) -> str:
    normalized = normalize_email(email)
    if normalized is None:
        raise ValueError("email must be valid")
    return f"email:{normalized}"


def _split_jwt(token: str) -> tuple[str, str, str]:
    parts = token.strip().split(".")
    if len(parts) != 3 or not all(parts):
        raise CloudflareAccessTokenError("Cloudflare Access JWT is malformed")
    return parts[0], parts[1], parts[2]


def _decode_json_segment(segment: str, label: str) -> Mapping[str, object]:
    try:
        raw = json.loads(_decode_base64url(segment))
    except Exception as exc:  # noqa: BLE001
        raise CloudflareAccessTokenError(f"Cloudflare Access JWT {label} is invalid") from exc
    if not isinstance(raw, dict):
        raise CloudflareAccessTokenError(f"Cloudflare Access JWT {label} must be an object")
    return raw


def _decode_base64url(value: str) -> bytes:
    padding_size = (-len(value)) % 4
    try:
        return base64.urlsafe_b64decode(f"{value}{'=' * padding_size}")
    except Exception as exc:  # noqa: BLE001
        raise CloudflareAccessTokenError("Cloudflare Access JWT base64 is invalid") from exc


def _numeric_date(value: object, claim: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CloudflareAccessTokenError(f"Cloudflare Access JWT {claim} is invalid")
    return float(value)


def _rsa_public_key(jwk: Mapping[str, object]) -> rsa.RSAPublicKey:
    if jwk.get("kty") != "RSA":
        raise CloudflareAccessTokenError("Cloudflare Access JWKS key type is invalid")
    exponent_raw = jwk.get("e")
    modulus_raw = jwk.get("n")
    if not isinstance(exponent_raw, str) or not isinstance(modulus_raw, str):
        raise CloudflareAccessTokenError("Cloudflare Access JWKS RSA key is invalid")
    exponent = int.from_bytes(_decode_base64url(exponent_raw), "big")
    modulus = int.from_bytes(_decode_base64url(modulus_raw), "big")
    try:
        return rsa.RSAPublicNumbers(exponent, modulus).public_key()
    except ValueError as exc:
        raise CloudflareAccessTokenError("Cloudflare Access JWKS RSA key is invalid") from exc
