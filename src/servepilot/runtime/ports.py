"""Local port allocation for engine replicas and internal servers."""

from __future__ import annotations

import socket
import threading

from servepilot.constants import DEFAULT_BACKEND_PORT_END, DEFAULT_BACKEND_PORT_START
from servepilot.exceptions import RuntimeStateError


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """True when nothing is listening on ``host:port`` (also checks the wildcard address)."""
    for bind_host in {host, "0.0.0.0"}:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except OSError:
                return False
    return True


class PortAllocator:
    """Hands out free ports from a range and remembers reservations until released.

    Reservations are logical: the port is checked for availability at allocation time and
    excluded from later allocations until :meth:`release` is called, which prevents two
    replicas launched in the same second from racing for one port.
    """

    def __init__(
        self,
        start: int = DEFAULT_BACKEND_PORT_START,
        end: int = DEFAULT_BACKEND_PORT_END,
        host: str = "127.0.0.1",
    ) -> None:
        if end < start:
            raise ValueError("port range end must be >= start")
        self._start = start
        self._end = end
        self._host = host
        self._reserved: set[int] = set()
        self._lock = threading.Lock()
        self._cursor = start

    def allocate(self) -> int:
        with self._lock:
            span = self._end - self._start + 1
            for _ in range(span):
                port = self._cursor
                self._cursor = self._start if self._cursor >= self._end else self._cursor + 1
                if port in self._reserved:
                    continue
                if port_is_free(port, self._host):
                    self._reserved.add(port)
                    return port
        raise RuntimeStateError(
            f"no free port available in range {self._start}-{self._end}",
            hints=["Set SERVEPILOT_BACKEND_PORT_START/END to a different range or free the ports."],
        )

    def allocate_many(self, count: int) -> list[int]:
        ports = [self.allocate() for _ in range(count)]
        return ports

    def release(self, port: int) -> None:
        with self._lock:
            self._reserved.discard(port)

    def release_all(self) -> None:
        with self._lock:
            self._reserved.clear()

    @property
    def reserved(self) -> set[int]:
        return set(self._reserved)


def ephemeral_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for a currently free ephemeral port (used for internal tuning routers)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
