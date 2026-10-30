from __future__ import annotations


def test_capture_observes_original_chunks_before_smaller_backend_view_limit():
    from shared.tool_capture import MAX_CAPTURE_BYTES, RawProcessCapture

    capture = RawProcessCapture()
    capture.feed("stdout", b" \xff\r\n")
    capture.feed("stderr", b"err\x00\n")
    capture.feed("stdout", b"a" * MAX_CAPTURE_BYTES)
    capture.feed("stdout", b"")
    observed = capture.observation(exit_code=None, timed_out=False, capture_error=None)
    assert observed.stdout == (b" \xff\r\n" + b"a" * MAX_CAPTURE_BYTES)[:MAX_CAPTURE_BYTES]
    assert observed.stdout_seen == MAX_CAPTURE_BYTES + 4
    assert observed.stderr == b"err\x00\n"
    assert observed.stderr_seen == 5
    assert observed.stdout_eof and not observed.stderr_eof
    assert not observed.stdout_complete and not observed.stderr_complete


def test_backend_collector_captures_read_bytes_before_its_existing_output_guard():
    import os
    from types import SimpleNamespace

    from core.workspace_python_execution import _collect
    from shared.tool_capture import RawProcessCapture

    read_out, write_out = os.pipe()
    read_err, write_err = os.pipe()
    os.write(write_out, b"0123456789abcdef")
    os.close(write_out)
    os.close(write_err)
    process = SimpleNamespace(
        stdout=os.fdopen(read_out, "rb"),
        stderr=os.fdopen(read_err, "rb"),
        returncode=0,
        poll=lambda: 0,
        wait=lambda **kwargs: 0,
    )
    capture = RawProcessCapture()
    try:
        result = _collect(process, timeout=1, limit=4, capture=capture)
    finally:
        process.stdout.close()
        process.stderr.close()
    assert result.stdout == b"0123"
    assert result.reason == "workspace_output_limit"
    observed = capture.observation(exit_code=None, timed_out=False, capture_error=None)
    assert observed.stdout == b"0123456789abcdef"
    assert observed.stdout_seen == 16
    assert not observed.stdout_eof and not observed.stdout_complete
