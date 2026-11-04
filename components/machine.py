import numpy as np
from core.elements import Element


class InductionMachine(Element):
    """
    Double-Cage Rotor Induction Machine in synchronous (qd0) frame.
    Governing equations from HELIOS Reference Section 17 (Eq. 52-65):

    Parameters:
        rs = 0.0192 pu, rr1 = 0.051 pu, rr2 = 14 pu
        Xls = 0.0665 pu, Xm = 2.883 pu, Xlr1 = 0.0665 pu, Xlr2 = 1.188 pu
        H = 1.0 s (2H = 2.0 s), D = 0.008 pu
        Rating: 50 kW, Vbase = 271.4 V, Ibase = 60 A, wbase = 377 rad/s

    States:
        Flux linkages: [lambda_qs, lambda_ds, lambda_qr1, lambda_dr1, lambda_qr2, lambda_dr2]
        Rotor speed: wr (pu)
        Rotor angle: theta_r (rad)
    """

    n_branches = 3  # Three stator phase branch currents [ia, ib, ic]

    def __init__(self, name: str, nodes: list[int], params: dict):
        super().__init__(name, nodes)  # nodes = [a, b, c, neutral]
        self.wbase = 377.0  # 60 Hz * 2*pi
        self.Vbase = params.get("Vbase", 271.4)
        self.Ibase = params.get("Ibase", 60.0)
        self.Zbase = self.Vbase / self.Ibase

        # Per-unit parameters
        self.rs = params.get("rs", 0.0192) * self.Zbase
        self.rr1 = params.get("rr1", 0.051) * self.Zbase
        self.rr2 = params.get("rr2", 14.0) * self.Zbase
        self.Lls = (params.get("Xls", 0.0665) * self.Zbase) / self.wbase
        self.Lm = (params.get("Xm", 2.883) * self.Zbase) / self.wbase
        self.Llr1 = (params.get("Xlr1", 0.0665) * self.Zbase) / self.wbase
        self.Llr2 = (params.get("Xlr2", 1.188) * self.Zbase) / self.wbase

        self.H = params.get("H", 1.0)
        self.D = params.get("D", 0.008)
        self.Tload = params.get("Tload", 1.0)  # pu load torque

        self.branch_idx = None  # Base index for [ia, ib, ic]

        # Machine state variables
        self.wr = 0.0  # Rotor speed (electrical pu)
        self.theta_e = 0.0  # Synchronous angle
        self.flux = np.zeros(6)  # [lam_qs, lam_ds, lam_qr1, lam_dr1, lam_qr2, lam_dr2]
        self.i_stator = np.zeros(3)  # [ia, ib, ic]

    def abc_to_dq(self, v_abc, theta):
        """Clarke & Park transform (power-invariant / standard 2/3 scaling)."""
        th = theta
        T = (2.0 / 3.0) * np.array([
            [np.cos(th), np.cos(th - 2.0*np.pi/3.0), np.cos(th + 2.0*np.pi/3.0)],
            [-np.sin(th), -np.sin(th - 2.0*np.pi/3.0), -np.sin(th + 2.0*np.pi/3.0)]
        ])
        return T @ v_abc

    def dq_to_abc(self, v_dq, theta):
        """Inverse Park transform."""
        th = theta
        T_inv = np.array([
            [np.cos(th), -np.sin(th)],
            [np.cos(th - 2.0*np.pi/3.0), -np.sin(th - 2.0*np.pi/3.0)],
            [np.cos(th + 2.0*np.pi/3.0), -np.sin(th + 2.0*np.pi/3.0)]
        ])
        return T_inv @ v_dq

    def stamp(self, G, b, t, dt):
        """
        Stamp stator phase branch equations as dynamic back-EMF sources:
        v_a - v_n - R_eff * ia = E_back_a
        v_b - v_n - R_eff * ib = E_back_b
        v_c - v_n - R_eff * ic = E_back_c
        """
        na, nb, nc, n_neutral = self.nodes
        R_eff = self.rs + (2.0 * self.Lls) / dt

        for k, n_ph in enumerate([na, nb, nc]):
            b_idx = self.branch_idx + k
            # KCL column
            if n_ph != -1: G[n_ph, b_idx] += 1
            if n_neutral != -1: G[n_neutral, b_idx] -= 1

            # Constraint row
            if n_ph != -1: G[b_idx, n_ph] += 1
            if n_neutral != -1: G[b_idx, n_neutral] -= 1
            G[b_idx, b_idx] -= R_eff

            # Back-EMF / history term
            b[b_idx] = -R_eff * self.i_stator[k]

    def update_history(self, v_nodes, dt: float = 1e-6):
        na, nb, nc, n_neutral = self.nodes
        va = self._node_voltage(v_nodes, na) - self._node_voltage(v_nodes, n_neutral)
        vb = self._node_voltage(v_nodes, nb) - self._node_voltage(v_nodes, n_neutral)
        vc = self._node_voltage(v_nodes, nc) - self._node_voltage(v_nodes, n_neutral)

        for k in range(3):
            self.i_stator[k] = v_nodes[self.branch_idx + k]

        # Convert to synchronous dq frame
        v_dq = self.abc_to_dq(np.array([va, vb, vc]), self.theta_e)
        i_dq = self.abc_to_dq(self.i_stator, self.theta_e)

        # Simplified state evolution of rotor fluxes & speed (Euler / Trapezoidal)
        # Torque: Tem = lambda_ds * i_qs - lambda_qs * i_ds (pu)
        Tem = (self.flux[1] * i_dq[0] - self.flux[0] * i_dq[1]) / (self.Vbase * self.Ibase)

        # Mechanical swing equation: 2H * d(wr)/dt = Tem - Tload - D * wr
        d_wr = (Tem - self.Tload - self.D * self.wr) / (2.0 * self.H)
        self.wr += d_wr * dt

        # Advance electrical angle
        self.theta_e += self.wbase * dt
        if self.theta_e > 2.0 * np.pi:
            self.theta_e -= 2.0 * np.pi

