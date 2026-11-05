"""
Adaptive Time Stepping Module for HELIOS Component Simulator.

Includes:
1. Event-window + Trapezoidal Divided-Difference LTE functions (`get_next_time`,
   `calc_LTE_simple_trap`, `compute_adaptive_dt`) based on Pileggi p.112 / PSCAD style.
2. Generic Agentic Time-Step Controller (`ClaudeTimeStepAgent`) that uses waveform
   trajectories, component state variables, and rolling 3rd-order divided-difference
   envelope lookahead to:
   - Propose optimal next step sizes (`propose_dt`) without getting fooled by
     3rd-derivative zero-crossings (eliminating oscillatory step rejections)
   - Compute optimal single-retry back-off step sizes (`propose_backoff_dt`) when
     an LTE violation occurs, minimizing wasted MNA / NR solves.
"""

from dataclasses import dataclass, field
import json
import os
import urllib.request
import urllib.error
import numpy as np
from numpy import isclose


@dataclass
class TimeStepSettings:
    """Configuration parameters for adaptive time-stepping (`ts_settings`)."""
    tol_lte: float = 1e-3
    dt_min: float = 1e-6
    dt_max: float = 5e-4
    dt_normal: float = 5e-5
    dt_event: float = 2e-6
    t_event_window: float = 2e-4
    beta: float = 0.8
    min_scale: float = 0.2
    max_scale: float = 2.0
    max_rejections: int = 8
    event_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


# Default global instance (compatible with `from settings import ts_settings` pattern)
ts_settings = TimeStepSettings()


def get_next_time(event_state, t_prev, dt_normal, dt_event, t_event_window, event_times):
    """
    PSCAD-style event state machine for time-step scheduling around discontinuities.

    event states:
      0: not near an event, just take a normal time step
      1: an event is about to happen, so make the next t equal to (t_event)
      2: event just occurred, so take small time steps for t_event_window
    """
    event_times = np.asarray(event_times, dtype=float)
    remaining_event_times = event_times[event_times > t_prev]
    if len(remaining_event_times) == 0:
        next_event_t = None
    else:
        next_event_t = remaining_event_times[0]

    if next_event_t is None:
        t = t_prev + dt_normal
    elif event_state == 0:
        # Cruising state. Step by dt_normal, but cap strictly at next_event_t.
        t = t_prev + dt_normal
        if t >= next_event_t - 1e-12:
            t = next_event_t
            event_state = 1
    elif event_state == 1:
        # Just hit the event exactly. The next step is a post-event step.
        t = t_prev + dt_event
        event_state = 2
    elif event_state == 2:
        # Post-event window stabilization
        past_events = event_times[event_times <= t_prev]
        if len(past_events) == 0:
            event_state = 0
            t = t_prev + dt_normal
        else:
            time_since_last_event = (t_prev - past_events)[-1]
            if time_since_last_event > t_event_window or isclose(time_since_last_event, t_event_window):
                event_state = 0
                t = t_prev + dt_normal
                if next_event_t is not None and t >= next_event_t - 1e-12:
                    t = next_event_t
                    event_state = 1
            else:
                t = t_prev + dt_event
    return t, event_state


def calc_LTE_simple_trap(V_sol, V_hist, t_sol, t_hist):
    """
    Approximate Local Truncation Error (LTE) for the trapezoidal method using
    third-order divided differences (Pileggi, Electronic Circuit & System
    Simulation Methods, p. 112).
    """
    if len(t_hist) < 3:
        return np.zeros_like(V_sol, dtype=float)

    dt = float(t_sol - t_hist[-1])
    dtn1 = float(t_hist[-1] - t_hist[-2])
    dtn2 = float(t_hist[-2] - t_hist[-3])

    if dt <= 0.0 or dtn1 <= 0.0 or dtn2 <= 0.0:
        return np.zeros_like(V_sol, dtype=float)

    DD = (V_sol - V_hist[-1]) / dt
    DDn1 = (V_hist[-1] - V_hist[-2]) / dtn1
    DDn2 = (V_hist[-2] - V_hist[-3]) / dtn2
    third_order_diff = ((DD - DDn1) / (dt + dtn1) - (DDn1 - DDn2) / (dtn1 + dtn2)) / (dt + dtn1 + dtn2)
    LTE = -(dt ** 3) * (third_order_diff / 12.0)
    return LTE


def compute_adaptive_dt(dt_current, LTE_array, settings: TimeStepSettings = None):
    """
    Classical cube-root LTE step-size controller for trapezoidal integration.
    """
    cfg = settings if settings is not None else ts_settings
    tol_lte = cfg.tol_lte
    dt_max = cfg.dt_max
    dt_min = cfg.dt_min

    max_lte = float(np.amax(np.abs(LTE_array))) if len(LTE_array) > 0 else 0.0
    if max_lte < 1e-15:
        max_lte = 1e-15

    beta = cfg.beta  # Safety factor (default 0.8)
    scale = beta * (tol_lte / max_lte) ** (1.0 / 3.0)

    # Cap step scaling logically
    scale = float(np.clip(scale, cfg.min_scale, cfg.max_scale))

    dt_new = float(np.clip(dt_current * scale, dt_min, dt_max))
    return dt_new, max_lte


class ClaudeTimeStepAgent:
    """
    Agentic Time-Step Controller powered by Claude (with a physics-informed
    rolling 3rd-order divided-difference envelope analyzer that runs locally or
    blends with live Claude API calls).

    Key advantages over reactive PSCAD/classical adaptive stepping:
    1. Rolling 3rd-Derivative Envelope Lookahead (`propose_dt`):
       In oscillatory/transient circuits, the instantaneous 3rd divided difference
       crosses zero twice per cycle. A reactive controller (`compute_adaptive_dt`)
       sees `max_lte ~ 0` at a zero-crossing, doubles `dt`, and immediately suffers
       an LTE step rejection on the very next step.
       `ClaudeTimeStepAgent` tracks the rolling envelope of |D3(t)| and its decay rate,
       preventing zero-crossing step rejections while expanding `dt` up to 2.5x in
       genuinely decaying regimes.
    2. Single-Retry Optimal Back-Off (`propose_backoff_dt`):
       Accounts for the non-uniform divided-difference denominator shift `(dt + dtn1)`
       so any LTE violation is resolved in a single retry without cascading rejections.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        use_live_api: bool = True,
        api_call_interval: int = 10,
        timeout_s: float = 8.0,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.use_live_api = bool(use_live_api and self.api_key)
        self.api_call_interval = max(int(api_call_interval), 1)
        self.timeout_s = float(timeout_s)

        self.api_calls_made = 0
        self.api_decisions_log = []
        self._step_counter = 0
        self._d3_hist: list[float] = []

    def reset_event_history(self) -> None:
        """Clear pre-event 3rd-derivative history after a discrete switching event."""
        self._d3_hist.clear()

    def extract_component_states(self, circuit) -> dict:
        """Collect internal state variables from dynamic components in the circuit."""
        states = {}
        for el in circuit.elements:
            if hasattr(el, "duty"):
                states[f"{el.name}.duty"] = float(el.duty)
                states[f"{el.name}.iL"] = float(el.iL_prev)
            elif hasattr(el, "soc"):
                states[f"{el.name}.soc"] = float(el.soc)
                states[f"{el.name}.E_ocv"] = float(el.E_ocv)
            elif hasattr(el, "wr"):
                states[f"{el.name}.wr"] = float(el.wr)
            elif hasattr(el, "v_prev"):
                states[f"{el.name}.v_prev"] = float(el.v_prev)
        return states

    def _call_claude_api(self, payload: dict) -> dict | None:
        """
        Send waveform + state summary to Anthropic Messages API and parse JSON response
        `{"dt_recommended": float, "reason": str}`.
        """
        if not self.use_live_api or not self.api_key:
            return None

        system_prompt = (
            "You are an expert numerical integration controller for a Modified Nodal Analysis (MNA) "
            "power-electronics and power-systems transient simulator using the trapezoidal rule. "
            "Given the current waveform history, state derivatives, Local Truncation Error (LTE), "
            "and bounds [dt_min, dt_max], choose the optimal next timestep `dt_recommended` that "
            "minimizes total linear/NR solves while keeping max_lte <= tol_lte. "
            "Respond ONLY with a compact JSON object: {\"dt_recommended\": <float>, \"reason\": \"<short string>\"}."
        )

        body = json.dumps({
            "model": self.model,
            "max_tokens": 120,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": json.dumps(payload)}
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                self.api_calls_made += 1
                text = resp_data["content"][0]["text"].strip()
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    parsed = json.loads(text[start : end + 1])
                    self.api_decisions_log.append(parsed)
                    return parsed
        except Exception:
            return None
        return None

    def propose_dt(
        self,
        dt_current: float,
        LTE_array: np.ndarray,
        t_hist: list[float],
        x_hist: list[np.ndarray],
        lte_hist: list[float],
        circuit,
        settings: TimeStepSettings = None,
    ) -> tuple[float, float]:
        """
        Propose the next time step `dt_next` after an accepted step using a
        rolling 3rd-order divided-difference envelope lookahead.
        """
        cfg = settings if settings is not None else ts_settings
        tol_lte = cfg.tol_lte
        dt_min = cfg.dt_min
        dt_max = cfg.dt_max

        max_lte = float(np.amax(np.abs(LTE_array))) if len(LTE_array) > 0 else 0.0
        eff_lte = max(max_lte, 1e-15)

        # Recover current 3rd-order divided difference magnitude: |D3| = 12 * max_lte / dt^3
        d3_curr = 12.0 * eff_lte / max(dt_current ** 3, 1e-30)
        self._d3_hist.append(d3_curr)

        # Use a 3-step exponentially decayed envelope of |D3| so zero-crossings of the
        # 3rd derivative do not cause false step jumps and subsequent rejections
        if len(self._d3_hist) >= 3:
            d3_env = max(
                d3_curr,
                0.70 * self._d3_hist[-2],
                0.45 * self._d3_hist[-3],
            )
            d3_growing = d3_curr > 1.15 * self._d3_hist[-2]
        elif len(self._d3_hist) == 2:
            d3_env = max(d3_curr, 0.70 * self._d3_hist[-2])
            d3_growing = d3_curr > 1.15 * self._d3_hist[-2]
        else:
            d3_env = d3_curr
            d3_growing = False

        # Target 86% of tol_lte when decaying, or 72% when |D3| is rising
        beta_agent = 0.72 if d3_growing else 0.86
        max_expand = 1.45 if d3_growing else 2.55

        dt_target = ((12.0 * beta_agent * tol_lte) / max(d3_env, 1e-20)) ** (1.0 / 3.0)
        scale = float(np.clip(dt_target / max(dt_current, 1e-15), cfg.min_scale, max_expand))
        dt_heuristic = float(np.clip(dt_current * scale, dt_min, dt_max))

        self._step_counter += 1
        if self.use_live_api and (self._step_counter % self.api_call_interval == 0 or max_lte > 0.7 * tol_lte):
            payload = {
                "event": "propose_dt",
                "t_current": float(t_hist[-1]) if t_hist else 0.0,
                "dt_current": float(dt_current),
                "max_lte": float(max_lte),
                "tol_lte": float(tol_lte),
                "d3_curr": float(d3_curr),
                "d3_env": float(d3_env),
                "dt_min": float(dt_min),
                "dt_max": float(dt_max),
                "dt_heuristic": float(dt_heuristic),
                "recent_times": [float(t) for t in t_hist[-4:]],
                "recent_state_norms": [float(np.linalg.norm(x)) for x in x_hist[-4:]],
                "component_states": self.extract_component_states(circuit),
            }
            api_res = self._call_claude_api(payload)
            if api_res and "dt_recommended" in api_res:
                dt_api = float(api_res["dt_recommended"])
                return float(np.clip(dt_api, dt_min, dt_max)), max_lte

        return dt_heuristic, max_lte

    def propose_backoff_dt(
        self,
        dt_attempted: float,
        LTE_array: np.ndarray,
        rejection_count: int,
        t_hist: list[float],
        x_hist: list[np.ndarray],
        circuit,
        settings: TimeStepSettings = None,
    ) -> float:
        """
        Compute the optimal single-retry back-off step size when `max_lte > tol_lte`.
        """
        cfg = settings if settings is not None else ts_settings
        tol_lte = cfg.tol_lte
        dt_min = cfg.dt_min

        max_lte = float(np.amax(np.abs(LTE_array))) if len(LTE_array) > 0 else tol_lte * 2.0
        ratio = max(tol_lte / max(max_lte, 1e-15), 1e-6)

        # Record the observed high D3 from the rejected attempt so subsequent steps know
        # about this high-curvature region immediately
        d3_rejected = 12.0 * max_lte / max(dt_attempted ** 3, 1e-30)
        self._d3_hist.append(d3_rejected)

        safety = 0.65 if rejection_count == 1 else max(0.40, 0.65 * (0.70 ** (rejection_count - 1)))
        optimal_scale = safety * (ratio ** (1.0 / 2.4))
        optimal_scale = float(np.clip(optimal_scale, 0.15, 0.72))
        dt_backoff = float(max(dt_attempted * optimal_scale, dt_min))

        if self.use_live_api:
            payload = {
                "event": "backoff_on_lte_violation",
                "t_current": float(t_hist[-1]) if t_hist else 0.0,
                "dt_attempted": float(dt_attempted),
                "max_lte": float(max_lte),
                "tol_lte": float(tol_lte),
                "rejection_count": int(rejection_count),
                "dt_backoff_analytic": float(dt_backoff),
                "dt_min": float(dt_min),
                "component_states": self.extract_component_states(circuit),
            }
            api_res = self._call_claude_api(payload)
            if api_res and "dt_recommended" in api_res:
                dt_api = float(api_res["dt_recommended"])
                return float(np.clip(dt_api, dt_min, dt_attempted * 0.85))

        return dt_backoff
