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

    Unknowns: the primary current i1 is ONE extra unknown, so this element
    owns one extra row/column of the augmented matrix (self.branch_idx,
    assigned by Circuit, same idea as VoltageSource).

    Sign convention: i1 flows from the p+ node into the transformer. The
    ideal part has v_x = a * v_s and i2 = -a * i1, where i2 is the current
    flowing from the s+ node into the transformer.

    KCL columns (rows are node rows, column is branch_idx). These come from
    "current leaving the node into the transformer":
        row p+ : +1
        row p- : -1
        row s+ : -a
        row s- : +a

    Constraint row (row = branch_idx), continuous form:
        (v_p+ - v_p-) - a (v_s+ - v_s-) = R i1 + L di1/dt
    Node-voltage coefficients in this row:
        p+ : +1,  p- : -1,  s+ : -a,  s- : +a

    TODO (derive on paper first):
      1. Let vL = L di1/dt. Apply the trapezoidal rule and solve for
         vL(n+1) in terms of i1(n+1), i1(n) and vL(n). It is the same rule
         you used for the inductor, rearranged so the current is the unknown.
      2. Substitute into the constraint row. Collect the terms containing
         i1(n+1) into one coefficient (the branch-column entry of the row),
         and move everything known (i1(n), vL(n)) to the right-hand side.
         That gives the coefficient and b[branch_idx].
      3. History for the next step: i1(n+1) is x[branch_idx]. vL(n+1)
         follows from the constraint equation after the solve.
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
        # dt: handle exactly as you did for Capacitor and Inductor

    def stamp(self, G, b, t, dt):
        # TODO: stamp the four KCL entries (skip ground, index -1)
        # TODO: stamp the four node-voltage entries of the constraint row
        # TODO: stamp the branch-column coefficient of the constraint row
        # TODO: set b[self.branch_idx] from the history terms
        p_plus, p_minus = self.nodes[0], self.nodes[1]
        s_plus, s_minus = self.nodes[2], self.nodes[3]

        # col is self.branch_idx
        if p_plus != -1:
            G[p_plus, self.branch_idx] += 1

        if p_minus != -1:
            G[p_minus, self.branch_idx] -= 1

        if s_plus != -1:
            G[s_plus, self.branch_idx] -= self.a

        if s_minus != -1:
            G[s_minus, self.branch_idx] += self.a

        #node-voltage entries in constraint row
        if p_plus != -1:
            G[self.branch_idx, p_plus] += 1
        if p_minus != -1:
            G[self.branch_idx, p_minus] -= 1
        if s_plus != -1:
            G[self.branch_idx, s_plus] -= self.a
        if s_minus != -1:
            G[self.branch_idx, s_minus] += self.a
        
        #branch-col coefficient
        G[self.branch_idx, self.branch_idx] -= (self.R + (2*self.L)/dt)

        #RHS historical terms set the res matrix
        b[self.branch_idx] = -(((2*self.L)/dt)*self.i_prev + self.vL_prev)
    




        



    def update_history(self, v_nodes):
        # TODO: i_prev  <- x[self.branch_idx]
        # TODO: vL_prev <- (v_p+ - v_p-) - a (v_s+ - v_s-) - R * i1
        #       use self._node_voltage() so ground nodes read as 0
        self.i_prev = v_nodes[self.branch_idx]
        p_plus, p_minus, s_plus, s_minus = self.nodes
        v_p_plus = self._node_voltage(v_nodes, p_plus)
        v_p_minus = self._node_voltage(v_nodes, p_minus)
        v_s_plus = self._node_voltage(v_nodes, s_plus)
        v_s_minus = self._node_voltage(v_nodes, s_minus)
        self.vL_prev = (v_p_plus - v_p_minus) - self.a*(v_s_plus - v_s_minus) - (self.R * self.i_prev) 
        