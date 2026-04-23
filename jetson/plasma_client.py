"""HTTP client for the ESP32 Plasma Gun firmware (v3.3).

The ESP32 hosts a SoftAP "PlasmaGun-AP" (password: plasma1234) at 192.168.4.1.
Connect the Jetson to that AP (or set the ESP32 to STA mode on your LAN
and update PLASMA_HOST below).

Endpoints used:
    GET /fire     - start fire sequence (returns "Sequence started!" or "Busy"/"Cooling down...")
    GET /status   - {"firing": bool, "step": str}
    GET /settings?pf=...&gf=...&...   - update timing params (locked from Jetson)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

log = logging.getLogger("plasma_client")


@dataclass
class PlasmaResponse:
    ok: bool
    text: str
    busy: bool = False
    cooldown: bool = False


class PlasmaClient:
    def __init__(self, host: str = "192.168.4.1", timeout: float = 2.0):
        self.base = f"http://{host}"
        self.timeout = timeout
        self.last_fire_at: float = 0.0
        self.last_status: dict = {"firing": False, "step": "UNKNOWN"}

    # ── Fire control ─────────────────────────────────────────────────────────
    def fire(self) -> PlasmaResponse:
        """Trigger a single full fire cycle. Non-blocking on the ESP32 side."""
        try:
            r = requests.get(f"{self.base}/fire", timeout=self.timeout)
            text = r.text.strip()
            busy = text.lower().startswith("busy")
            cooldown = "cooldown" in text.lower()
            ok = r.status_code == 200 and not busy and not cooldown
            if ok:
                self.last_fire_at = time.time()
            log.info("FIRE -> %s", text)
            return PlasmaResponse(ok=ok, text=text, busy=busy, cooldown=cooldown)
        except requests.RequestException as e:
            log.warning("FIRE failed: %s", e)
            return PlasmaResponse(ok=False, text=f"connection error: {e}")

    def status(self) -> dict:
        try:
            r = requests.get(f"{self.base}/status", timeout=self.timeout)
            self.last_status = r.json()
            return self.last_status
        except (requests.RequestException, ValueError) as e:
            log.debug("status failed: %s", e)
            self.last_status = {"firing": False, "step": "OFFLINE", "error": str(e)}
            return self.last_status

    # ── Settings (lock plasma params from Jetson at startup) ─────────────────
    def lock_settings(self,
                      pre_flush_ms: int = 15000,
                      gas_fill_ms: int = 225,
                      gas_pulses: int = 1,
                      gas_pause_ms: int = 200,
                      settle_ms: int = 0,
                      spark_ms: int = 500,
                      spark_retries: int = 3,
                      spark_gap_ms: int = 300,
                      exhaust_ms: int = 15000,
                      cooldown_ms: int = 5000) -> bool:
        params = dict(
            pf=pre_flush_ms, gf=gas_fill_ms, gp=gas_pulses, gg=gas_pause_ms,
            sd=settle_ms, sk=spark_ms, sr=spark_retries, sg=spark_gap_ms,
            ex=exhaust_ms, cd=cooldown_ms,
        )
        try:
            r = requests.get(f"{self.base}/settings", params=params, timeout=self.timeout)
            log.info("Locked settings: %s", r.text.strip())
            return r.status_code == 200
        except requests.RequestException as e:
            log.warning("lock_settings failed: %s", e)
            return False

    # ── Manual component test (used for diagnostics from web UI) ─────────────
    def test(self, component: str, state: str) -> bool:
        """component: valve|spark|fan|all   state: on|off"""
        try:
            r = requests.get(f"{self.base}/test", params={"c": component, "s": state},
                             timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False
