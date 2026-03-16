from __future__ import annotations

from core.mwv.models import VerificationResult, VerificationStatus


def extract_verifier_excerpt(
    result: VerificationResult,
    *,
    max_lines: int = 3,
    max_chars: int = 300,
) -> str:
    if result.error:
        return _trim_excerpt(result.error, max_chars=max_chars)
    stderr_excerpt = _excerpt_from_text(result.stderr, max_lines=max_lines, max_chars=max_chars)
    if stderr_excerpt:
        return stderr_excerpt
    stdout_excerpt = _excerpt_from_text(result.stdout, max_lines=max_lines, max_chars=max_chars)
    if stdout_excerpt:
        return stdout_excerpt
    if result.exit_code is None:
        if result.status == VerificationStatus.ERROR:
            return "Verifier завершился с ошибкой без подробностей."
        return "Неизвестная ошибка проверки."
    return f"exit_code={result.exit_code}"


def summarize_verifier_failure(
    result: VerificationResult,
    *,
    max_chars: int = 200,
) -> str:
    if result.status == VerificationStatus.ERROR:
        if result.error:
            return result.error
        return "Ошибка верификации"
    excerpt = _excerpt_from_text(result.stderr, max_lines=2, max_chars=max_chars)
    if excerpt:
        return excerpt
    excerpt = _excerpt_from_text(result.stdout, max_lines=2, max_chars=max_chars)
    if excerpt:
        return excerpt
    if result.exit_code is None:
        return "Неизвестная ошибка проверки"
    return f"Код возврата: {result.exit_code}"


def verifier_fail_type(result: VerificationResult) -> str:
    if result.error:
        return "runtime_error"
    if result.stderr.strip():
        return "stderr"
    if result.stdout.strip():
        return "stdout"
    if result.exit_code is None:
        return "empty"
    return "exit_code"


def _excerpt_from_text(text: str, *, max_lines: int, max_chars: int) -> str:
    normalized_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_noise_line(line):
            continue
        normalized_lines.append(line)
        if len(normalized_lines) >= max_lines:
            break
    if not normalized_lines:
        return ""
    return _trim_excerpt("\n".join(normalized_lines), max_chars=max_chars)


def _is_noise_line(line: str) -> bool:
    return bool(line) and all(char in {"=", "-", "_", "*", " "} for char in line)


def _trim_excerpt(text: str, *, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3].rstrip()}..."
