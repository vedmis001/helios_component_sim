import numpy as np

from core.elements import Element


def leakage_from_pu(x_pu: float, r_pu: float, v_base_primary: float,
                    s_base: float, f_base: float = 60.0):
    """
    Convert per-unit leakage values to (R, L) in ohms and henries, referred
    to the primary side.

    v_base_primary: primary rated voltage that matches s_base. For a 3-phase
                    bank use line-to-line volts with the 3-phase MVA base;
                    that gives the per-phase base impedance directly.
    s_base:         VA base.
    """
    z_base = v_base_primary ** 2 / s_base
    omega = 2.0 * np.pi * f_base
    R = r_pu * z_base
    L = x_pu * z_base / omega
    return R, L


class Transformer(Element):
    """
    Single-phase, reduced-order transformer: an ideal ratio with the leakage
    R and L in series on the primary side. No magnetizing branch, no
    saturation.

        p+ --[ R -- L ]--( ideal  a:1 )-- s+
        p- ---------------------------- s-

    nodes  = [p_plus, p_minus, s_plus, s_minus]
    params = {"ratio": a, "R": R_leak, "L": L_leak}
             a = primary rated voltage / secondary rated voltage
             R, L are referred to the primary side (see leakage_from_pu)
    """

    n_branches = 1

    def __init__(self, name: str, nodes: list[int], params: dict):
        super().__init__(name, nodes)
        self.a = params["ratio"]
        self.R = params["R"]
        self.L = params["L"]
        self.branch_idx = None   # set by Circuit before the first stamp
        self.i_prev = 0.0        # primary current at the previous step
        self.vL_prev = 0.0       # voltage across the leakage inductance, previous step

    def stamp(self, G, b, t, dt):
        p_plus, p_minus = self.nodes[0], self.nodes[1]
        s_plus, s_minus = self.nodes[2], self.nodes[3]
        r_eq = self.R + (2 * self.L) / dt

        if hasattr(G, "add"):
            if p_plus != -1:
                G.add(p_plus, self.branch_idx, 1.0)
                G.add(self.branch_idx, p_plus, 1.0)
            if p_minus != -1:
                G.add(p_minus, self.branch_idx, -1.0)
                G.add(self.branch_idx, p_minus, -1.0)
            if s_plus != -1:
                G.add(s_plus, self.branch_idx, -self.a)
                G.add(self.branch_idx, s_plus, -self.a)
            if s_minus != -1:
                G.add(s_minus, self.branch_idx, self.a)
                G.add(self.branch_idx, s_minus, self.a)
            G.add(self.branch_idx, self.branch_idx, -r_eq)
        else:
            if p_plus != -1:
                G[p_plus, self.branch_idx] += 1
                G[self.branch_idx, p_plus] += 1
            if p_minus != -1:
                G[p_minus, self.branch_idx] -= 1
                G[self.branch_idx, p_minus] -= 1
            if s_plus != -1:
                G[s_plus, self.branch_idx] -= self.a
                G[self.branch_idx, s_plus] -= self.a
            if s_minus != -1:
                G[s_minus, self.branch_idx] += self.a
                G[self.branch_idx, s_minus] += self.a
            G[self.branch_idx, self.branch_idx] -= r_eq

        b[self.branch_idx] = -(((2 * self.L) / dt) * self.i_prev + self.vL_prev)

    def update_history(self, v_nodes):
        self.i_prev = v_nodes[self.branch_idx]
        p_plus, p_minus, s_plus, s_minus = self.nodes
        v_p_plus = self._node_voltage(v_nodes, p_plus)
        v_p_minus = self._node_voltage(v_nodes, p_minus)
        v_s_plus = self._node_voltage(v_nodes, s_plus)
        v_s_minus = self._node_voltage(v_nodes, s_minus)
        self.vL_prev = (v_p_plus - v_p_minus) - self.a * (v_s_plus - v_s_minus) - (self.R * self.i_prev)
