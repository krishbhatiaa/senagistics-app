"""
SenaGistics - Security & Audit layer.

Implements the three security pillars described in the architecture slide:
  1. Encryption at rest  -> every log entry's payload is encrypted (Fernet/AES).
  2. RBAC                -> simple role-based access control for actions.
  3. Tamper-evident audit trail -> each entry is hash-chained to the previous
     one (like a mini blockchain), so editing or deleting a past entry breaks
     the chain and is detectable.

This is a real, runnable implementation (not a slide-only claim) kept
intentionally simple so it's easy to explain in a viva.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

from cryptography.fernet import Fernet

AUDIT_LOG_PATH = "audit_log.jsonl"
KEY_PATH = "audit_key.key"

# ---------------------------------------------------------------------------
# Role-Based Access Control
# ---------------------------------------------------------------------------

# Each role is allowed to perform a subset of actions.
ROLE_PERMISSIONS = {
    "depot_clerk": {"log_consumption", "view_own_depot"},
    "duty_officer": {"log_consumption", "view_own_depot", "view_all_depots", "acknowledge_anomaly"},
    "logistics_command": {
        "log_consumption", "view_own_depot", "view_all_depots",
        "acknowledge_anomaly", "approve_procurement", "view_audit_log",
    },
}

ROLES = list(ROLE_PERMISSIONS.keys())


def is_authorized(role: str, action: str) -> bool:
    return action in ROLE_PERMISSIONS.get(role, set())


# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------

def get_or_create_key(path: str = KEY_PATH) -> bytes:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(path, "wb") as f:
        f.write(key)
    return key


# ---------------------------------------------------------------------------
# Hash-chained, encrypted audit trail
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    index: int
    timestamp: float
    role: str
    user: str
    action: str
    encrypted_payload: str  # base64 fernet token, stored as str
    prev_hash: str
    entry_hash: str = ""

    def compute_hash(self) -> str:
        raw = f"{self.index}|{self.timestamp}|{self.role}|{self.user}|{self.action}|{self.encrypted_payload}|{self.prev_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()


class AuditLog:
    """A simple append-only, hash-chained, encrypted audit log backed by a
    JSON-lines file. Every entry's hash depends on the previous entry's hash,
    so tampering with any past entry invalidates every hash after it —
    this is what makes the trail "tamper-evident"."""

    def __init__(self, path: str = AUDIT_LOG_PATH, key: Optional[bytes] = None):
        self.path = path
        self.key = key or get_or_create_key()
        self.fernet = Fernet(self.key)

    def _read_all(self):
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _last_hash(self) -> str:
        entries = self._read_all()
        if not entries:
            return "GENESIS"
        return entries[-1]["entry_hash"]

    def append(self, role: str, user: str, action: str, payload: dict) -> AuditEntry:
        """Encrypt + append a new audit entry. Raises PermissionError if the
        role is not authorized to perform this action (RBAC enforcement)."""
        if not is_authorized(role, action):
            raise PermissionError(f"Role '{role}' is not authorized to perform '{action}'")

        entries = self._read_all()
        index = len(entries)
        prev_hash = self._last_hash()
        timestamp = time.time()

        plaintext = json.dumps(payload).encode()
        encrypted_payload = self.fernet.encrypt(plaintext).decode()

        entry = AuditEntry(
            index=index,
            timestamp=timestamp,
            role=role,
            user=user,
            action=action,
            encrypted_payload=encrypted_payload,
            prev_hash=prev_hash,
        )
        entry.entry_hash = entry.compute_hash()

        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

        return entry

    def decrypt_payload(self, encrypted_payload: str) -> dict:
        plaintext = self.fernet.decrypt(encrypted_payload.encode())
        return json.loads(plaintext)

    def verify_chain(self):
        """Walk the entire log and verify every hash link. Returns
        (is_valid: bool, broken_at_index: int | None)."""
        entries = self._read_all()
        prev_hash = "GENESIS"
        for e in entries:
            expected = AuditEntry(
                index=e["index"], timestamp=e["timestamp"], role=e["role"],
                user=e["user"], action=e["action"],
                encrypted_payload=e["encrypted_payload"], prev_hash=prev_hash,
            ).compute_hash()
            if e["prev_hash"] != prev_hash or e["entry_hash"] != expected:
                return False, e["index"]
            prev_hash = e["entry_hash"]
        return True, None

    def all_entries_decrypted(self):
        """Return all entries with their payloads decrypted, for display."""
        out = []
        for e in self._read_all():
            payload = self.decrypt_payload(e["encrypted_payload"])
            out.append({**e, "payload": payload})
        return out


if __name__ == "__main__":
    # quick self-test
    if os.path.exists(AUDIT_LOG_PATH):
        os.remove(AUDIT_LOG_PATH)
    if os.path.exists(KEY_PATH):
        os.remove(KEY_PATH)

    log = AuditLog()
    log.append("depot_clerk", "clerk_01", "log_consumption", {"depot": "Depot-Alpha", "fuel_litres": 512.3})
    log.append("duty_officer", "officer_07", "acknowledge_anomaly", {"depot": "Depot-Alpha", "note": "verified, false alarm"})

    try:
        log.append("depot_clerk", "clerk_01", "approve_procurement", {"depot": "Depot-Alpha"})
    except PermissionError as e:
        print("RBAC correctly blocked unauthorized action:", e)

    valid, broken_at = log.verify_chain()
    print("Chain valid:", valid)

    for entry in log.all_entries_decrypted():
        print(entry["index"], entry["role"], entry["action"], "->", entry["payload"])
