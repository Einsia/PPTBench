from __future__ import annotations

import json
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
import random
import re
import ssl
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi


class ResponseTooLargeError(OSError):
    pass


class CachedHttpClient:
    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        timeout: int = 60,
        min_interval: float = 1.0,
        max_attempts: int = 4,
        max_response_bytes: int | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.max_response_bytes = max_response_bytes
        self._last_request_at = 0.0
        self._request_lock = threading.Lock()
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    def _wait(self, interval: float | None = None) -> None:
        with self._request_lock:
            minimum = self.min_interval if interval is None else interval
            remaining = minimum - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()

    def get(self, url: str, cache_key: str, suffix: str = ".bin") -> bytes:
        cache_path = self.cache_dir / f"{cache_key}{suffix}"
        if cache_path.exists() and cache_path.stat().st_size:
            if self.max_response_bytes and cache_path.stat().st_size > self.max_response_bytes:
                raise ResponseTooLargeError(
                    f"cached response is {cache_path.stat().st_size} bytes; "
                    f"limit is {self.max_response_bytes}"
                )
            return cache_path.read_bytes()
        partial_path = cache_path.with_name(cache_path.name + ".part")

        error: Exception | None = None
        for attempt in range(self.max_attempts):
            offset = partial_path.stat().st_size if partial_path.exists() else 0
            self._wait(min(self.min_interval, 0.25) if offset else self.min_interval)
            headers = {"User-Agent": self.user_agent, "Accept": "*/*"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(
                url,
                headers=headers,
            )
            try:
                with urlopen(
                    request, timeout=self.timeout, context=self._ssl_context
                ) as response:
                    status = getattr(response, "status", response.getcode())
                    append = bool(offset and status == 206)
                    expected = int(response.headers.get("Content-Length", "0") or 0)
                    content_range = response.headers.get("Content-Range", "")
                    total_match = re.search(r"/(\d+)$", content_range)
                    total_size = int(total_match.group(1)) if total_match else 0
                    declared_size = total_size or (offset + expected if append else expected)
                    if self.max_response_bytes and declared_size > self.max_response_bytes:
                        partial_path.unlink(missing_ok=True)
                        raise ResponseTooLargeError(
                            f"response is {declared_size} bytes; "
                            f"limit is {self.max_response_bytes}"
                        )
                    written = 0
                    with partial_path.open("ab" if append else "wb") as output:
                        while chunk := response.read(1_048_576):
                            output.write(chunk)
                            written += len(chunk)
                    if expected and written < expected:
                        raise IncompleteRead(b"", expected - written)
                    if total_size and partial_path.stat().st_size < total_size:
                        raise IncompleteRead(
                            b"", total_size - partial_path.stat().st_size
                        )
                partial_path.replace(cache_path)
                return cache_path.read_bytes()
            except HTTPError as exc:
                error = exc
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise
                retry_after = exc.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(15, 2**attempt)
                )
            except ResponseTooLargeError:
                raise
            except IncompleteRead as exc:
                error = exc
                delay = 0
            except (URLError, TimeoutError, RemoteDisconnected) as exc:
                error = exc
                delay = min(15, 2**attempt)
            time.sleep(delay + random.random() * 0.25)

        if error is None:
            # This is only reachable if max_attempts is zero or a future retry
            # branch forgets to retain its exception.  Keep the failure explicit
            # so ``python -O`` cannot turn it into an unrelated UnboundLocalError.
            raise OSError(
                f"request failed after {self.max_attempts} attempts without an exception"
            )
        raise OSError(f"request failed after {self.max_attempts} attempts: {error}") from error

    def save_json(self, name: str, value: object) -> None:
        path = self.cache_dir / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
