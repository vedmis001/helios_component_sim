import time as _time
import numpy as np

from .circuit import Circuit
from .adaptive import (
    TimeStepSettings,
    ts_settings,
    get_next_time,
    calc_LTE_simple_trap,
    compute_adaptive_dt,
    ClaudeTimeStepAgent,
)


class Simulator:
    """
    Time-domain MNA driver supporting:
      1. Fixed-timestep simulation (`mode="fixed"`, default)
      2. PSCAD-style Event + Trapezoidal LTE adaptive stepping (`mode="pscad"`)
      3. Claude Agentic adaptive stepping (`mode="claude"`)

    Results:
        self.time  shape (n_steps + 1,)
        self.x     shape (n_steps + 1, n_unknowns); row k is the full
                   solution vector at self.time[k]. Row 0 is t = 0.
        self.stats dictionary of solver performance & LTE telemetry
    """

    def __init__(
        self,
        circuit: Circuit,
        dt: float,
        t_end: float,
        mode: str = "fixed",
        settings: TimeStepSettings | None = None,
        agent: ClaudeTimeStepAgent | None = None,
    ):
        self.circuit = circuit
        self.dt = float(dt)
        self.t_end = float(t_end)
        self.mode = mode.lower()
        self.settings = settings
        self.agent = agent

        self.n_steps = int(round(self.t_end / self.dt))
        self.time = np.zeros(self.n_steps + 1)
        self.x = np.zeros((self.n_steps + 1, circuit.n_unknowns))

        self.dt_history = np.full(self.n_steps, self.dt)
        self.lte_history = np.zeros(self.n_steps)
        self.stats = {}

    def run(self, mode: str | None = None) -> dict:
        """
        Execute the simulation in the selected mode ('fixed', 'pscad', or 'claude').
        Defaults to `self.mode` ('fixed' if not specified).
        """
        active_mode = (mode or self.mode).lower()
        if active_mode == "fixed":
            return self._run_fixed()
        elif active_mode in ("pscad", "claude", "adaptive"):
            return self.run_adaptive(mode=active_mode, settings=self.settings, agent=self.agent)
        else:
            raise ValueError(f"Unknown simulation mode: {active_mode!r}")

    def _run_fixed(self) -> dict:
        """
        Fixed-timestep time loop.
        """
        t0 = _time.perf_counter()
        self.n_steps = int(round(self.t_end / self.dt))
        self.time = np.zeros(self.n_steps + 1)
        self.x = np.zeros((self.n_steps + 1, self.circuit.n_unknowns))
        self.dt_history = np.full(self.n_steps, self.dt)
        self.lte_history = np.zeros(self.n_steps)

        t_hist = [0.0]
        x_hist = [self.x[0].copy()]

        for step in range(1, self.n_steps + 1):
            t_new = step * self.dt
            (G, b) = self.circuit.build_system(t_new, self.dt)
            x_new = self.circuit.solve(G, b)
            self.time[step] = t_new
            self.x[step] = x_new
            self.circuit.update_histories(x_new, self.dt)

            if len(t_hist) >= 3:
                lte_vec = calc_LTE_simple_trap(x_new, x_hist[-3:], t_new, t_hist[-3:])
                self.lte_history[step - 1] = float(np.amax(np.abs(lte_vec)))
            t_hist.append(t_new)
            x_hist.append(x_new)

        wall_s = _time.perf_counter() - t0
        density_info = self.circuit.density_summary(self.t_end, self.dt)
        self.stats = {
            "mode": "fixed",
            "n_accepted_steps": self.n_steps,
            "n_rejected_steps": 0,
            "total_solves": self.n_steps,
            "wall_time_s": wall_s,
            "dt_min_used": self.dt,
            "dt_max_used": self.dt,
            "dt_mean_used": self.dt,
            "max_lte": float(np.max(self.lte_history)) if len(self.lte_history) > 0 else 0.0,
            "matrix_density": density_info["density"],
            "matrix_sparsity": density_info["sparsity"],
            "matrix_nnz": density_info["nnz"],
            "triplet_capacity": density_info["allocated_capacity"],
        }
        return self.stats

    def run_adaptive(
        self,
        mode: str = "pscad",
        settings: TimeStepSettings | None = None,
        event_times: np.ndarray | list[float] | None = None,
        agent: ClaudeTimeStepAgent | None = None,
    ) -> dict:
        """
        Adaptive time-stepping loop supporting:
          - mode='pscad': Event state machine (`get_next_time`) + Pileggi 3rd-order
            divided-difference LTE (`calc_LTE_simple_trap`) + classical step controller
            (`compute_adaptive_dt`) with LTE violation step rejection.
          - mode='claude': Agentic controller (`ClaudeTimeStepAgent`) that inspects
            waveform history, derivatives, component states, and LTE trends to choose
            both forward steps and single-retry back-off steps on LTE violations.
        """
        t0 = _time.perf_counter()
        mode = mode.lower()
        if mode == "adaptive":
            mode = "pscad"

        cfg = settings if settings is not None else (self.settings or TimeStepSettings(
            tol_lte=ts_settings.tol_lte,
            dt_min=min(self.dt * 0.2, ts_settings.dt_min),
            dt_max=max(self.dt * 25.0, ts_settings.dt_max),
            dt_normal=self.dt * 2.0,
            dt_event=self.dt,
            t_event_window=max(10.0 * self.dt, ts_settings.t_event_window),
        ))

        ev_times = (
            np.sort(np.asarray(event_times, dtype=float))
            if event_times is not None
            else np.sort(np.asarray(cfg.event_times, dtype=float))
        )

        if mode == "claude":
            active_agent = agent or self.agent or ClaudeTimeStepAgent(use_live_api=True)
        else:
            active_agent = None

        t_hist: list[float] = [0.0]
        x_hist: list[np.ndarray] = [np.zeros(self.circuit.n_unknowns, dtype=float)]
        dt_hist: list[float] = []
        lte_hist: list[float] = []

        t_prev = 0.0
        dt_curr = float(np.clip(self.dt, cfg.dt_min, cfg.dt_max))
        event_state = 0
        post_switch_steps = 10

        n_accepted = 0
        n_rejected = 0
        total_solves = 0

        while t_prev < self.t_end - 1e-14:
            # First 2 steps warm up the 3-point history required for 3rd-order divided differences
            if len(t_hist) < 3:
                dt_cand = min(dt_curr, cfg.dt_event, self.t_end - t_prev)
                t_cand = t_prev + dt_cand
                next_event_state = event_state
            else:
                if mode == "pscad":
                    t_cand, next_event_state = get_next_time(
                        event_state,
                        t_prev,
                        dt_curr,
                        cfg.dt_event,
                        cfg.t_event_window,
                        ev_times,
                    )
                else:
                    # Claude agentic mode: uses event lookahead to hit exact event_times,
                    # and uses a compact 3-step post-event window to re-establish post-switch
                    # 3rd-order divided differences before handing control back to the agent
                    eff_window = 2.5 * cfg.dt_event
                    t_cand, next_event_state = get_next_time(
                        event_state,
                        t_prev,
                        dt_curr,
                        cfg.dt_event,
                        eff_window,
                        ev_times,
                    )

                if t_cand > self.t_end:
                    t_cand = self.t_end
                dt_cand = t_cand - t_prev
                if dt_cand <= 1e-15:
                    break

            # A switching transition occurs when stepping out of t_event (event_state == 1 -> 2)
            crossing_switch = (event_state == 1 and next_event_state == 2)

            # Inner LTE verification & back-off loop (counts every MNA / NR solve)
            rejection_count = 0
            while True:
                (G, b) = self.circuit.build_system(t_cand, dt_cand)
                x_cand = self.circuit.solve(G, b)
                total_solves += 1

                # Do not compute 3rd-order divided differences across a switching discontinuity
                if len(t_hist) < 3 or crossing_switch or post_switch_steps < 2:
                    lte_vec = np.zeros_like(x_cand)
                    max_lte = 0.0
                    break

                # Evaluate LTE on the circuit's node voltages (first n_nodes unknowns)
                n_v = self.circuit.n_nodes
                v_hist_slice = [xh[:n_v] for xh in x_hist[-3:]]
                lte_vec = calc_LTE_simple_trap(x_cand[:n_v], v_hist_slice, t_cand, t_hist[-3:])
                max_lte = float(np.amax(np.abs(lte_vec))) if len(lte_vec) > 0 else 0.0

                # Check if LTE tolerance is violated and we can still back off
                if (
                    max_lte > cfg.tol_lte
                    and dt_cand > cfg.dt_min * 1.001
                    and rejection_count < cfg.max_rejections
                ):
                    n_rejected += 1
                    rejection_count += 1

                    if mode == "claude" and active_agent is not None:
                        dt_cand = active_agent.propose_backoff_dt(
                            dt_attempted=dt_cand,
                            LTE_array=lte_vec,
                            rejection_count=rejection_count,
                            t_hist=t_hist,
                            x_hist=x_hist,
                            circuit=self.circuit,
                            settings=cfg,
                        )
                    else:
                        # Standard PSCAD/classical back-off on LTE violation
                        dt_cand, _ = compute_adaptive_dt(dt_cand, lte_vec, settings=cfg)
                        dt_cand = min(dt_cand, dt_curr * 0.7)
                        dt_cand = max(dt_cand, cfg.dt_min)

                    t_cand = min(t_prev + dt_cand, self.t_end)
                    next_event_state = 0 if event_state == 0 else event_state
                    continue

                # Step accepted!
                break

            # Commit accepted step
            self.circuit.update_histories(x_cand, dt_cand)
            if crossing_switch:
                post_switch_steps = 0
                if active_agent is not None:
                    active_agent.reset_event_history()
            else:
                post_switch_steps += 1

            t_prev = t_cand
            event_state = next_event_state
            n_accepted += 1

            t_hist.append(t_cand)
            x_hist.append(x_cand)
            dt_hist.append(dt_cand)
            lte_hist.append(max_lte)

            # Propose next dt for cruising steps
            if len(t_hist) >= 3 and post_switch_steps >= 2:
                if mode == "claude" and active_agent is not None:
                    dt_curr, _ = active_agent.propose_dt(
                        dt_current=dt_cand,
                        LTE_array=lte_vec,
                        t_hist=t_hist,
                        x_hist=x_hist,
                        lte_hist=lte_hist,
                        circuit=self.circuit,
                        settings=cfg,
                    )
                else:
                    dt_curr, _ = compute_adaptive_dt(dt_cand, lte_vec, settings=cfg)
            elif post_switch_steps < 2:
                dt_curr = cfg.dt_event

        wall_s = _time.perf_counter() - t0
        self.time = np.asarray(t_hist, dtype=float)
        self.x = np.vstack(x_hist)
        self.n_steps = n_accepted
        self.dt_history = np.asarray(dt_hist, dtype=float)
        self.lte_history = np.asarray(lte_hist, dtype=float)

        density_info = self.circuit.density_summary(self.t_end, self.dt)
        self.stats = {
            "mode": mode,
            "n_accepted_steps": n_accepted,
            "n_rejected_steps": n_rejected,
            "total_solves": total_solves,
            "wall_time_s": wall_s,
            "dt_min_used": float(np.min(self.dt_history)) if len(self.dt_history) > 0 else self.dt,
            "dt_max_used": float(np.max(self.dt_history)) if len(self.dt_history) > 0 else self.dt,
            "dt_mean_used": float(np.mean(self.dt_history)) if len(self.dt_history) > 0 else self.dt,
            "max_lte": float(np.max(self.lte_history)) if len(self.lte_history) > 0 else 0.0,
            "tol_lte": cfg.tol_lte,
            "matrix_density": density_info["density"],
            "matrix_sparsity": density_info["sparsity"],
            "matrix_nnz": density_info["nnz"],
            "triplet_capacity": density_info["allocated_capacity"],
            "claude_api_calls": active_agent.api_calls_made if active_agent is not None else 0,
        }
        return self.stats

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------
    def voltage(self, node_name: str) -> np.ndarray:
        """Voltage vs time at a node. Ground is identically zero."""
        idx = self.circuit.node_index(node_name)
        if idx == -1:
            return np.zeros_like(self.time)
        return self.x[:, idx]

    def branch_current(self, source_name: str) -> np.ndarray:
        """Current vs time through a voltage source (MNA sign convention)."""
        return self.x[:, self.circuit.vsource_index(source_name)]
