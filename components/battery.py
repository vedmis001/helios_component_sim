import numpy as np
from core.elements import Element


class Battery(Element):
    """
    Shepherd battery model (0.6 kV pack, Q = 0.367 kAh, SOC0 = 80%).
    Governing equations from HELIOS Reference Section 14 (Eq. 36-44):

    Coulomb counting:
        it(t) = it_prev + ibatt * dt / 3600   (Ah extracted)
        SOC(t) = SOC0 - (100 / Q) * it(t)

    Terminal voltage (discharge vs. charge):
        V_disch = E0 - K*(Q/(Q - it))*i* - K*(Q/(Q - it))*it + A*exp(-B*it) - R*ibatt
        V_chg   = E0 - K*(Q/(it + 0.1*Q))*i* - K*(Q/(Q - it))*it + A*exp(-B*it) - R*ibatt

    MNA Representation:
        Stamped as a dynamic Thevenin equivalent between [p_plus, p_minus]:
        - Internal open-circuit voltage E_batt(t) in series with internal resistance R.
        - Uses 1 extra branch current variable (self.branch_idx), like VoltageSource.
    """

    n_branches = 1

    def __init__(self, name: str, nodes: list[int], params: dict):
        super().__init__(name, nodes)
        self.Vnom = params.get("Vnom", 600.0)      # V
        self.Q = params.get("Q", 0.367)            # kAh = 367 Ah
        self.SOC0 = params.get("SOC0", 80.0)       # %
        
        # Per-unit parameters extracted in Eq. (40)-(43)
        self.A = params.get("A", 0.12 * self.Vnom)
        self.B = params.get("B", 7.5 / self.Q)
        self.K = params.get("K", 1.6e-3 * self.Vnom)
        self.R = params.get("R", 0.01 * (self.Vnom / 1000.0))  # approx R_int
        self.E0 = params.get("E0", 1.042 * self.Vnom)
        self.tau_filter = params.get("tau_filter", 0.05)       # filter for i*

        self.branch_idx = None

        # State variables
        self.it = (1.0 - self.SOC0 / 100.0) * self.Q   # Ah already extracted
        self.soc = self.SOC0
        self.i_filt = 0.0                              # i* low-pass filtered current
        self.ibatt_prev = 0.0
        self.E_ocv = self.Vnom

    def stamp(self, G, b, t, dt):
        """
        Stamp branch equation: (v_pos - v_neg) - R * ibatt = E_ocv
        KCL: current ibatt leaves p_plus, enters p_minus.
        """
        p_plus, p_minus = self.nodes[0], self.nodes[1]

        # KCL column entries
        if p_plus != -1:
            G[p_plus, self.branch_idx] += 1
        if p_minus != -1:
            G[p_minus, self.branch_idx] -= 1

        # Constraint row entries
        if p_plus != -1:
            G[self.branch_idx, p_plus] += 1
        if p_minus != -1:
            G[self.branch_idx, p_minus] -= 1

        # Internal resistance drop in constraint row
        G[self.branch_idx, self.branch_idx] -= self.R

        # Open circuit voltage on RHS
        b[self.branch_idx] = self.E_ocv

    def update_history(self, v_nodes, dt: float = 1e-6):
        # Current leaving battery into circuit
        ibatt = v_nodes[self.branch_idx]
        self.ibatt_prev = ibatt

        # Filtered current i* (to avoid algebraic loops)
        alpha = dt / (self.tau_filter + dt)
        self.i_filt += alpha * (ibatt - self.i_filt)

        # Coulomb counting (Ah)
        # ibatt > 0 discharges, ibatt < 0 charges
        self.it += (ibatt * dt) / 3600.0
        self.it = np.clip(self.it, 0.0, 0.999 * self.Q)
        self.soc = self.SOC0 - (100.0 / self.Q) * self.it

        # Recompute open-circuit voltage for next step
        it_val = max(1e-4, self.it)
        exp_term = self.A * np.exp(-self.B * it_val)

        if ibatt >= 0:
            # Discharge equation (38)
            self.E_ocv = (self.E0 
                          - self.K * (self.Q / (self.Q - it_val)) * self.i_filt 
                          - self.K * (self.Q / (self.Q - it_val)) * it_val 
                          + exp_term)
        else:
            # Charge equation (39)
            self.E_ocv = (self.E0 
                          - self.K * (self.Q / (it_val + 0.1 * self.Q)) * self.i_filt 
                          - self.K * (self.Q / (self.Q - it_val)) * it_val 
                          + exp_term)

