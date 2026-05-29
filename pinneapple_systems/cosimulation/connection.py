"""Connection dataclass for co-simulation graphs.

A Connection is a directed edge from one node's output port to another
node's input port.  Port addresses use the dotted string format::

    "node_name.port_name"

Example::

    conn = Connection.parse("spring.force", "mass.F_ext")
    print(conn)   # spring.force -> mass.F_ext
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Connection:
    """Immutable directed edge between two node ports.

    Attributes:
        src_node:  name of the source node.
        src_port:  output port on the source node.
        dst_node:  name of the destination node.
        dst_port:  input port on the destination node.
    """

    src_node: str
    src_port: str
    dst_node: str
    dst_port: str

    # ------------------------------------------------------------------
    @classmethod
    def parse(cls, src: str, dst: str) -> "Connection":
        """Build a Connection from two 'node.port' strings.

        Args:
            src: source address, e.g. ``"spring.force"``
            dst: destination address, e.g. ``"mass.F_ext"``

        Raises:
            ValueError: if either string is not in 'node.port' format.
        """
        src_node, src_port = _split_port(src)
        dst_node, dst_port = _split_port(dst)
        return cls(src_node, src_port, dst_node, dst_port)

    # ------------------------------------------------------------------
    def src_address(self) -> str:
        """Return 'src_node.src_port' string."""
        return f"{self.src_node}.{self.src_port}"

    def dst_address(self) -> str:
        """Return 'dst_node.dst_port' string."""
        return f"{self.dst_node}.{self.dst_port}"

    def __str__(self) -> str:
        return f"{self.src_node}.{self.src_port} -> {self.dst_node}.{self.dst_port}"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _split_port(s: str) -> Tuple[str, str]:
    """Split 'node.port' into (node, port).

    Raises:
        ValueError: if the string is malformed.
    """
    parts = s.rsplit(".", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(
            f"Invalid port address {s!r}. "
            "Expected format: 'node_name.port_name'"
        )
    return parts[0].strip(), parts[1].strip()
