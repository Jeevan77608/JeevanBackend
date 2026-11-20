"""
AES-256-GCM encryption for sensitive session data.

This module implements industry-standard encryption following:
- NIST SP 800-38D (GCM Mode)
- OWASP Cryptographic Storage Cheat Sheet
- Key rotation support via versioned ciphertext

Security Features:
- AES-256-GCM authenticated encryption
- 96-bit random nonce (NIST recommended)
- Version prefix for key rotation support
- AAD (Additional Authenticated Data) binding
- Constant-time operations where applicable
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class EncryptionError(Exception):
    """Base exception for encryption operations."""

    pass


class EncryptionKeyError(EncryptionError):
    """Invalid or missing encryption key."""

    pass


class DecryptionError(EncryptionError):
    """Failed to decrypt data (tampered, wrong key, or corrupted)."""

    pass


class EncryptionVersionError(EncryptionError):
    """Unsupported encryption version."""

    pass


# =============================================================================
# Constants
# =============================================================================


class EncryptionVersion(str, Enum):
    """Encryption version identifiers for key rotation support."""

    V1 = "v1"  # Current version: AES-256-GCM


# Current version used for new encryptions
CURRENT_VERSION: Final[EncryptionVersion] = EncryptionVersion.V1

# Version separator in encrypted output
VERSION_SEPARATOR: Final[str] = ":"

# GCM nonce size (96 bits = 12 bytes per NIST SP 800-38D)
NONCE_SIZE: Final[int] = 12

# AES-256 key size (256 bits = 32 bytes)
KEY_SIZE: Final[int] = 32


def _safe_b64decode(s: str) -> bytes:
    """
    Decode base64 string, adding padding if needed.

    Python's b64decode requires the number of data characters (excluding padding)
    to be a multiple of 4. Strip existing padding first, then add exactly the
    right amount so we never get 'number of data characters (85) cannot be
    1 more than a multiple of 4'.
    """
    if not s:
        raise ValueError("Empty string cannot be base64 decoded")
    s = s.strip()
    # Remove any existing padding so we count only data chars, then add correct amount
    data_part = s.rstrip("=")
    padding_needed = (4 - (len(data_part) % 4)) % 4
    padded = data_part + "=" * padding_needed
    return base64.b64decode(padded)


# =============================================================================
# Encryption Service Interface
# =============================================================================


class IEncryptionService(ABC):
    """Interface for encryption services."""

    @abstractmethod
    def encrypt(self, plaintext: str, associated_data: bytes | None = None) -> str:
        """Encrypt plaintext string."""
        ...

    @abstractmethod
    def decrypt(self, encrypted_data: str, associated_data: bytes | None = None) -> str:
        """Decrypt encrypted string."""
        ...

    @abstractmethod
    def encrypt_dict(
        self, data: dict[str, Any], associated_data: bytes | None = None
    ) -> str:
        """Encrypt dictionary as JSON."""
        ...

    @abstractmethod
    def decrypt_dict(
        self, encrypted_data: str, associated_data: bytes | None = None
    ) -> dict[str, Any]:
        """Decrypt to dictionary."""
        ...


# =============================================================================
# Key Container (Secure Key Handling)
# =============================================================================


@dataclass(frozen=True, slots=True)
class EncryptionKey:
    """
    Secure container for encryption key.

    Immutable and uses slots for memory efficiency.
    The key is stored in its raw bytes form.
    """

    key_bytes: bytes
    version: EncryptionVersion = EncryptionVersion.V1

    def __post_init__(self) -> None:
        """Validate key after initialization."""
        if len(self.key_bytes) != KEY_SIZE:
            raise EncryptionKeyError(
                f"Encryption key must be {KEY_SIZE} bytes ({KEY_SIZE * 8} bits), "
                f"got {len(self.key_bytes)} bytes"
            )

    @classmethod
    def from_base64(cls, encoded_key: str) -> EncryptionKey:
        """Create key from base64-encoded string. Pads to multiple of 4 if needed."""
        try:
            key_bytes = _safe_b64decode(encoded_key)
        except Exception as e:
            raise EncryptionKeyError(f"Invalid base64 encoding: {e}") from e
        return cls(key_bytes=key_bytes)

    def derive_key_id(self) -> str:
        """
        Derive a non-sensitive key identifier for logging.

        Uses first 8 chars of SHA-256 hash of the key.
        """
        key_hash = hashlib.sha256(self.key_bytes).hexdigest()
        return key_hash[:8]

    def __repr__(self) -> str:
        """Safe representation that doesn't expose key material."""
        return f"EncryptionKey(version={self.version}, key_id={self.derive_key_id()})"


# =============================================================================
# AES-256-GCM Encryption Service
# =============================================================================


class EncryptionService(IEncryptionService):
    """
    AES-256-GCM encryption service following industry best practices.

    Features:
    - Authenticated encryption (confidentiality + integrity)
    - Version prefix for future key rotation
    - AAD support for context binding
    - Secure random nonce generation

    Thread Safety:
    - This class is thread-safe. The AESGCM cipher can be used
      concurrently as each encrypt/decrypt call uses a unique nonce.
    """

    def __init__(
        self,
        encryption_key: str | EncryptionKey,
        *,
        version: EncryptionVersion = CURRENT_VERSION,
    ) -> None:
        """
        Initialize encryption service.

        Args:
            encryption_key: Base64-encoded 32-byte key or EncryptionKey instance
            version: Encryption version to use for new encryptions

        Raises:
            EncryptionKeyError: If key is invalid
        """
        if isinstance(encryption_key, str):
            self._key = EncryptionKey.from_base64(encryption_key)
        else:
            self._key = encryption_key

        self._version = version
        self._aesgcm = AESGCM(self._key.key_bytes)

        logger.debug(
            "Encryption service initialized",
            extra={"key_id": self._key.derive_key_id(), "version": version.value},
        )

    @property
    def key_id(self) -> str:
        """Get the key identifier for logging purposes."""
        return self._key.derive_key_id()

    def encrypt(self, plaintext: str, associated_data: bytes | None = None) -> str:
        """
        Encrypt plaintext using AES-256-GCM.

        The output format is: {version}:{base64(nonce + ciphertext + tag)}

        Args:
            plaintext: UTF-8 string to encrypt
            associated_data: Optional AAD for context binding (e.g., session ID)

        Returns:
            Versioned, base64-encoded encrypted string

        Raises:
            EncryptionError: If encryption fails
        """
        if not plaintext:
            raise EncryptionError("Cannot encrypt empty plaintext")

        try:
            # Generate cryptographically secure random nonce
            nonce = secrets.token_bytes(NONCE_SIZE)
            plaintext_bytes = plaintext.encode("utf-8")

            # AES-GCM encrypt (automatically appends 128-bit auth tag)
            ciphertext = self._aesgcm.encrypt(nonce, plaintext_bytes, associated_data)

            # Combine: nonce || ciphertext || tag
            combined = nonce + ciphertext
            encoded = base64.b64encode(combined).decode("utf-8")

            # Prepend version for future key rotation support
            return f"{self._version.value}{VERSION_SEPARATOR}{encoded}"

        except Exception as e:
            logger.error(
                "Encryption failed",
                extra={"key_id": self.key_id, "error": str(e)},
            )
            raise EncryptionError(f"Encryption failed: {e}") from e

    def decrypt(self, encrypted_data: str, associated_data: bytes | None = None) -> str:
        """
        Decrypt AES-256-GCM encrypted data.

        Supports versioned format: {version}:{base64_data}
        Also supports legacy format without version prefix.

        Args:
            encrypted_data: Versioned encrypted string
            associated_data: AAD used during encryption (must match exactly)

        Returns:
            Decrypted UTF-8 string

        Raises:
            DecryptionError: If decryption fails (wrong key, tampered, corrupted)
            EncryptionVersionError: If version is unsupported
        """
        if not encrypted_data:
            raise DecryptionError("Cannot decrypt empty data")

        try:
            # Parse version and payload
            version_str, payload = self._parse_encrypted_data(encrypted_data)

            # Validate version
            if version_str and version_str != self._version.value:
                raise EncryptionVersionError(
                    f"Unsupported encryption version: {version_str}"
                )

            # Decode base64 (pad to multiple of 4 if needed; some encoders omit padding)
            combined = _safe_b64decode(payload)

            # Validate minimum length (nonce + at least 1 byte + 16 byte tag)
            min_length = NONCE_SIZE + 1 + 16
            if len(combined) < min_length:
                raise DecryptionError("Encrypted data too short")

            # Extract nonce and ciphertext+tag
            nonce = combined[:NONCE_SIZE]
            ciphertext = combined[NONCE_SIZE:]

            # Decrypt and verify
            plaintext_bytes = self._aesgcm.decrypt(nonce, ciphertext, associated_data)
            return plaintext_bytes.decode("utf-8")

        except InvalidTag:
            logger.warning(
                "Decryption failed: authentication tag mismatch",
                extra={"key_id": self.key_id},
            )
            raise DecryptionError(
                "Decryption failed: data integrity check failed (tampered or wrong key)"
            )
        except (EncryptionVersionError, DecryptionError):
            raise
        except Exception as e:
            logger.warning(
                "Decryption failed",
                extra={"key_id": self.key_id, "error": str(e)},
            )
            raise DecryptionError(f"Decryption failed: {e}") from e

    def _parse_encrypted_data(self, encrypted_data: str) -> tuple[str | None, str]:
        """Parse version prefix from encrypted data."""
        if VERSION_SEPARATOR in encrypted_data:
            parts = encrypted_data.split(VERSION_SEPARATOR, 1)
            if len(parts) == 2 and parts[0] in [v.value for v in EncryptionVersion]:
                return parts[0], parts[1]
        # Legacy format without version
        return None, encrypted_data

    def encrypt_dict(
        self, data: dict[str, Any], associated_data: bytes | None = None
    ) -> str:
        """
        Encrypt a dictionary as JSON.

        Args:
            data: Dictionary to encrypt
            associated_data: Optional AAD for context binding

        Returns:
            Encrypted string
        """
        json_str = json.dumps(data, separators=(",", ":"), default=str)
        return self.encrypt(json_str, associated_data)

    def decrypt_dict(
        self, encrypted_data: str, associated_data: bytes | None = None
    ) -> dict[str, Any]:
        """
        Decrypt to dictionary.

        Args:
            encrypted_data: Encrypted JSON string
            associated_data: AAD used during encryption

        Returns:
            Decrypted dictionary
        """
        json_str = self.decrypt(encrypted_data, associated_data)
        return json.loads(json_str)


# =============================================================================
# Key Generation Utility
# =============================================================================


def generate_encryption_key() -> str:
    """
    Generate a cryptographically secure AES-256 encryption key.

    Returns:
        Base64-encoded 32-byte key suitable for AES-256

    Example:
        >>> key = generate_encryption_key()
        >>> print(f"ENCRYPTION_KEY={key}")
    """
    key_bytes = secrets.token_bytes(KEY_SIZE)
    return base64.b64encode(key_bytes).decode("utf-8")


def constant_time_compare(a: bytes, b: bytes) -> bool:
    """
    Compare two byte strings in constant time to prevent timing attacks.

    Args:
        a: First byte string
        b: Second byte string

    Returns:
        True if equal, False otherwise
    """
    return hmac.compare_digest(a, b)

