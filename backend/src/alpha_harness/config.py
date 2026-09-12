"""Application settings.

Values come from (in precedence order): process environment, the repo-root ``.env``,
then the defaults below.

**No BRAIN credentials here.** They are typed into the sign-in screen and sealed in the
local vault, and nothing reads them from the environment. Seeding them from a file meant
a checked-out repository could sign in as its owner, which is a backdoor rather than a
convenience — and it made the sign-in screen a formality that could silently be skipped.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# .../backend/src/alpha_harness/config.py -> .../alpha-harness
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration for the Alpha Harness backend."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="AH_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- BRAIN platform -------------------------------------------------
    brain_api_base: str = "https://api.worldquantbrain.com"

    # A consultant's daily simulation allowance. The platform only reveals the real
    # figure in the headers of a simulation POST, so this is what the Today page shows
    # until the first batch of the day comes back and replaces it with the truth.
    daily_simulation_allowance: int = 5000

    # --- Local storage --------------------------------------------------
    # Home-relative by default. Must stay on native ext4 — see CLAUDE.md.
    data_dir: Path = Path.home() / ".alpha-harness"

    # --- HTTP -----------------------------------------------------------
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Ceiling on a single poll loop, so a stuck server-side job cannot hang a
    # request forever. Simulations are polled by the background tracker, not here.
    poll_timeout_seconds: float = 300.0
    # BRAIN sometimes returns a very small Retry-After; never hammer faster than this.
    min_retry_after_seconds: float = 1.0

    # Minimum spacing between outbound BRAIN requests. A catalog crawl needs ~200
    # requests per scope (``/data-fields`` caps ``limit`` at 50, and USA/D1/TOP3000
    # alone holds 10,000 fields); unpaced, that earns a 429 within seconds. The
    # platform's own guidance is to keep request rates near what its web UI generates.
    #
    # Measured against the live platform: at 0.35s the crawl completed in 229s but was
    # throttled on roughly half its requests (96 retries over 200 pages). Since the
    # retry backoff was setting the real pace anyway, asking politely at ~1 req/s costs
    # no wall-clock time and stops hammering an endpoint that is telling us to slow down.
    min_request_interval_seconds: float = 1.0
    # Attempts for retryable failures (429 throttling, 503, transport errors).
    request_attempts: int = 6

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> list[str]:
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                import json
                try:
                    return json.loads(value)
                except Exception:
                    pass
            return [v.strip() for v in value.split(",") if v.strip()]
        return value  # type: ignore[return-value]

    @field_validator("data_dir", mode="after")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def sqlite_path(self) -> Path:
        """Operational state: credentials, sessions, simulation records, templates."""
        return self.data_dir / "harness.db"

    @property
    def duckdb_path(self) -> Path:
        """Analytical store: the data-field catalog."""
        return self.data_dir / "catalog.duckdb"

    @property
    def key_path(self) -> Path:
        """AES-GCM master key. Created 0600 on first run."""
        return self.data_dir / "key"

    def ensure_data_dir(self) -> None:
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
