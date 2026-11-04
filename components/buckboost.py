import numpy as np
from core.elements import Element


class BuckBoost(Element):
    """
    Averaged switch model of the bidirectional DC-DC converter.
    Governing equations from HELIOS Reference Section 14 (Eq. 23-35):

    Terminals:
        nodes = [VH_pos, VH_neg, VL_pos, VL_neg]
        VH: High-voltage side (DC link, 900 V)
        VL: Low-voltage side (Battery, 600 V)

    Inductor dynamics:
        Buck (charging, mode='buck'):
            L_tot * dIL/dt = D_Buck * VH - VL
            IH = D_Buck * IL
        Boost (discharging, mode='boost'):
            L_tot * dIL/dt = VL - (1 - D_Boost) * VH
            IH = (1 - D_Boost) * IL

    PI Controllers:
        Boost (voltage regulation):
            eV = V_H_ref - VH
            D_Boost = sat[0, Dmax_boost](Kp_boost * eV + Ki_boost * integral(eV))
        Buck (power regulation):
            eP = P_charge_ref - (VH * IH)
            D_Buck = sat[0, Dmax_buck](Kp_buck * eP + Ki_buck * integral(eP))
    """

    n_branches = 1  # Inductor current iL is an MNA branch variable

    def __init__(self, name: str, nodes: list[int], params: dict):
        super().__init__(name, nodes)
        self.Ltot = params.get("Ltot", 1.5e-3)       # Total inductance (H)
        self.mode = params.get("mode", "boost")      # 'buck' or 'boost'
        self.VH_ref = params.get("VH_ref", 900.0)    # DC link reference
        self.P_ref = params.get("P_ref", 100e3)      # Charge power target (W)
        
        # Controller gains from Section 14
        self.Dmax_boost = params.get("Dmax_boost", 0.70)
        self.Kp_boost = params.get("Kp_boost", 0.05)
        self.Ti_boost = params.get("Ti_boost", 0.02)
        
        self.Dmax_buck = params.get("Dmax_buck", 0.98)
        self.Kp_buck = params.get("Kp_buck", 0.5)
        self.Ti_buck = params.get("Ti_buck", 0.2)

        self.branch_idx = None
        self.duty = 0.5
        self.iL_prev = 0.0
        self.vL_ind_prev = 0.0
        self.err_integral = 0.0

    def set_mode(self, mode: str):
        if mode in ("buck", "boost"):
            self.mode = mode
            self.err_integral = 0.0

    def stamp(self, G, b, t, dt):
        """
        Stamp discretized inductor companion equation:
        v_ind(n+1) = (2*L/dt)*(iL(n+1) - iL(n)) - v_ind(n)
        Coupled with duty cycle scaling D into VH and VL nodes.
        """
        vh_p, vh_n, vl_p, vl_n = self.nodes
        R_L = (2.0 * self.Ltot) / dt

        if self.mode == "buck":
            eff_D = self.duty
            # v_ind = eff_D * VH - VL
            # Constraint row: eff_D * (vh_p - vh_n) - (vl_p - vl_n) - R_L * iL = RHS
            if vh_p != -1: G[self.branch_idx, vh_p] += eff_D
            if vh_n != -1: G[self.branch_idx, vh_n] -= eff_D
            if vl_p != -1: G[self.branch_idx, vl_p] -= 1.0
            if vl_n != -1: G[self.branch_idx, vl_n] += 1.0

            # KCL injections:
            # High side draws IH = eff_D * iL
            if vh_p != -1: G[vh_p, self.branch_idx] += eff_D
            if vh_n != -1: G[vh_n, self.branch_idx] -= eff_D
            # Low side receives IL
            if vl_p != -1: G[vl_p, self.branch_idx] -= 1.0
            if vl_n != -1: G[vl_n, self.branch_idx] += 1.0

        else: # boost
            D_off = 1.0 - self.duty
            # v_ind = VL - D_off * VH
            # Constraint row: (vl_p - vl_n) - D_off * (vh_p - vh_n) - R_L * iL = RHS
            if vl_p != -1: G[self.branch_idx, vl_p] += 1.0
            if vl_n != -1: G[self.branch_idx, vl_n] -= 1.0
            if vh_p != -1: G[self.branch_idx, vh_p] -= D_off
            if vh_n != -1: G[self.branch_idx, vh_n] += D_off

            # KCL injections:
            # Low side draws IL
            if vl_p != -1: G[vl_p, self.branch_idx] += 1.0
            if vl_n != -1: G[vl_n, self.branch_idx] -= 1.0
            # High side injects IH = D_off * iL
            if vh_p != -1: G[vh_p, self.branch_idx] -= D_off
            if vh_n != -1: G[vh_n, self.branch_idx] += D_off

        # Inductor numerical resistance on diagonal
        G[self.branch_idx, self.branch_idx] -= R_L
        # History term on RHS
        b[self.branch_idx] = -(R_L * self.iL_prev + self.vL_ind_prev)

    def update_history(self, v_nodes, dt: float = 1e-6):
        vh_p, vh_n, vl_p, vl_n = self.nodes
        vh = self._node_voltage(v_nodes, vh_p) - self._node_voltage(v_nodes, vh_n)
        vl = self._node_voltage(v_nodes, vl_p) - self._node_voltage(v_nodes, vl_n)

        # Current through the inductor
        iL = v_nodes[self.branch_idx]

        # Update inductor internal voltage across Ltot
        if self.mode == "buck":
            v_ind = self.duty * vh - vl
            # PI controller for charging power:
            p_actual = vh * (self.duty * iL)
            err = self.P_ref - p_actual
            self.err_integral += err * dt
            duty_raw = self.Kp_buck * err + (self.Kp_buck / self.Ti_buck) * self.err_integral
            self.duty = np.clip(duty_raw, 0.0, self.Dmax_buck)
        else:
            D_off = 1.0 - self.duty
            v_ind = vl - D_off * vh
            # PI controller for DC link voltage regulation:
            err = self.VH_ref - vh
            self.err_integral += err * dt
            duty_raw = self.Kp_boost * err + (self.Kp_boost / self.Ti_boost) * self.err_integral
            self.duty = np.clip(duty_raw, 0.0, self.Dmax_boost)

        self.iL_prev = iL
        self.vL_ind_prev = v_ind

