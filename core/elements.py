#R, L, C, V-source, I-source classes

import numpy as np
from abc import ABC, abstractmethod


class Element(ABC):
    """
    Base class for all circuit elements.
    Every element knows which nodes it connects and how to stamp itself
    into the MNA system Gx = b.
    
    Node index convention:
        -1 is ground (reference node, never appears in the matrix)
        0, 1, 2, ... are the actual matrix rows/columns
    """

    def __init__(self, name: str, nodes: list[int]):
        self.name = name
        self.nodes = nodes  # e.g. [from_node, to_node]

    @abstractmethod
    def stamp(self, G: np.ndarray, b: np.ndarray, t: float, dt: float) -> None:
        """
        Stamp this element's contribution into G and b.
        Called once per timestep before solving.
        G: conductance matrix (n_nodes x n_nodes)
        b: RHS vector (n_nodes,)
        t: current simulation time
        dt: timestep size
        """
        pass

    def update_history(self, v_nodes: np.ndarray) -> None:
        """
        Called once per timestep after solving.
        v_nodes: full solution vector x from this timestep.
        Dynamic elements (L, C) override this to update their
        history current for the next timestep.
        Static elements (R) don't need to.
        """
        pass

    def _node_voltage(self, v_nodes: np.ndarray, node_idx: int) -> float:
        """Helper: returns voltage at a node, 0.0 if ground (-1)."""
        if node_idx == -1:
            return 0.0
        return v_nodes[node_idx]


# ---------------------------------------------------------------------------
# Resistor
# ---------------------------------------------------------------------------

class Resistor(Element):
    """
    Stamps conductance G = 1/R between nodes[0] and nodes[1].
    
    MNA stamp:
        G[n+][n+] += g
        G[n-][n-] += g
        G[n+][n-] -= g
        G[n-][n+] -= g
    where g = 1/R. Ground nodes are skipped.
    """

    def __init__(self, name: str, nodes: list[int], R: float):
        super().__init__(name, nodes)
        self.R = R


    def stamp(self, G, b, t, dt):
        # implement the four stamp operations above
        # remember to skip any node that is -1 (ground)
        g_res = 1/self.R
        n_pos, n_neg = self.nodes[0], self.nodes[1]
        if n_pos != -1:
            G[n_pos, n_pos] += g_res
        if n_neg != -1:
            G[n_neg, n_neg] += g_res
        
        #off diagonals
        if (n_pos != -1) and (n_neg != -1):
            G[n_pos, n_neg] -= g_res
            G[n_neg, n_pos] -= g_res




# ---------------------------------------------------------------------------
# Capacitor
# ---------------------------------------------------------------------------

class Capacitor(Element):
    """
    Trapezoidal companion model:
        i(n+1) = G_eq * v(n+1) + I_hist
    where:
        G_eq   = 2C/dt
        I_hist = -G_eq * v(n) - i(n)   (updated each timestep)

    Stamps G_eq as a conductance, and I_hist as a current source into b.
    Current source convention: positive I_hist injects into the + node.
    """

    def __init__(self, name: str, nodes: list[int], C: float):
        super().__init__(name, nodes)
        self.C = C
        self.I_hist = 0.0   # history current, initialised to 0
        self.v_prev = 0.0   # voltage across element last timestep
        self.i_prev = 0.0   # current through element last timestep

    def stamp(self, G, b, t, dt):
        # TODO: compute G_eq from self.C and dt
        # TODO: stamp G_eq as a conductance (same pattern as Resistor)
        # TODO: inject self.I_hist into b:
        #   b[n+] -= I_hist   (current leaving + node)
        #   b[n-] += I_hist   (current entering - node)
        # Again skip ground nodes
        
        n_pos, n_neg = self.nodes[0], self.nodes[1]
        #calculate time-step dependent params
        self.geq = (2*self.C)/dt


        #add to conductance matrix
        if n_pos != -1:
            G[n_pos, n_pos] += self.geq
        if n_neg != -1:
            G[n_neg, n_neg] += self.geq

        #off diagonals
        if (n_pos != -1) and (n_neg != -1):
            G[n_pos, n_neg] -= self.geq
            G[n_neg, n_pos] -= self.geq

        #inject equiv i into b
        if n_pos != -1:
            b[n_pos] -= self.I_hist
        if n_neg != -1:
            b[n_neg] += self.I_hist

    def update_history(self, v_nodes):
        # TODO: read v_prev from v_nodes using self._node_voltage()
        # TODO: compute i_prev = G_eq * v_prev + I_hist  (current from last step)
        # TODO: recompute I_hist for next step

        #voltage across terminals
        v_prev = self._node_voltage(v_nodes, self.nodes[0]) - self._node_voltage(v_nodes, self.nodes[1])

        #current thru cap
        i_cap = (self.geq * v_prev) + self.I_hist

        #update I_hist for next step
        self.I_hist = (-self.geq * v_prev) - i_cap
        self.v_prev, self.i_prev = v_prev, i_cap





# ---------------------------------------------------------------------------
# Inductor
# ---------------------------------------------------------------------------

class Inductor(Element):
    """
    Trapezoidal companion model — derive this yourself on paper first.
    Same structure as Capacitor: G_eq conductance + I_hist current source,
    but the expressions for G_eq and I_hist are different.

    start from v = L * di/dt and apply the trapezoidal rule.
    """

    def __init__(self, name: str, nodes: list[int], L: float):
        super().__init__(name, nodes)
        self.L = L
        self.I_hist = 0.0
        self.v_prev = 0.0
        self.i_prev = 0.0

    def stamp(self, G, b, t, dt):
        # TODO: derive G_eq for an inductor (it is NOT 2L/dt)
        # TODO: stamp and inject history current — same pattern as Capacitor
        
        n_pos, n_neg = self.nodes[0], self.nodes[1]
        #calculate time-step dependent params
        self.geq = dt/(2*self.L)


        #add to conductance matrix
        if n_pos != -1:
            G[n_pos, n_pos] += self.geq
        if n_neg != -1:
            G[n_neg, n_neg] += self.geq

        #off diagonals
        if (n_pos != -1) and (n_neg != -1):
            G[n_pos, n_neg] -= self.geq
            G[n_neg, n_pos] -= self.geq

        #inject equiv i into b
        if n_pos != -1:
            b[n_pos] -= self.I_hist
        if n_neg != -1:
            b[n_neg] += self.I_hist

    

    def update_history(self, v_nodes):
        # TODO: same structure as Capacitor's update_history
        # but the formula for I_hist is different — derive it
                
        #voltage across terminals
        v_prev = self._node_voltage(v_nodes, self.nodes[0]) - self._node_voltage(v_nodes, self.nodes[1])

        #current thru cap
        i_ind = (self.geq * v_prev) + self.I_hist

        #update I_hist for next step
        self.I_hist = (self.geq * v_prev) + i_ind
        self.v_prev, self.i_prev = v_prev, i_ind



# ---------------------------------------------------------------------------
# VoltageSource
# ---------------------------------------------------------------------------

class VoltageSource(Element):
    """
    An ideal voltage source forces v[n+] - v[n-] = V(t).
    This introduces an unknown branch current, so the MNA matrix
    must be augmented by one row and one column.

    This is the trickiest stamp — read about 'Modified' in MNA before
    implementing. The element needs to know its own row index in the
    augmented matrix (stored as self.branch_idx, assigned by Circuit
    when the system is built).

    Stamp pattern (branch current variable = x[branch_idx]):
        G[n+][branch_idx] += 1
        G[n-][branch_idx] -= 1
        G[branch_idx][n+] += 1
        G[branch_idx][n-] -= 1
        b[branch_idx]      = V(t)
    """

    def __init__(self, name: str, nodes: list[int], V):
        super().__init__(name, nodes)
        # V can be a float (DC) or a callable V(t) (time-varying)
        self.V = V
        self.branch_idx = None  # set by Circuit before first stamp

    def voltage(self, t: float) -> float:
        if callable(self.V):
            return self.V(t)
        return float(self.V)

    def stamp(self, G, b, t, dt):
        # TODO: implement the 5-line stamp above
        # self.branch_idx is the row/col index for this source's
        # branch current in the augmented matrix
        # skip ground nodes as usual

        n_pos, n_neg = self.nodes[0], self.nodes[1]
        if n_pos != -1:
            G[n_pos][self.branch_idx] += 1
            G[self.branch_idx][n_pos] += 1
        if n_neg != -1:
            G[n_neg][self.branch_idx] -= 1
            G[self.branch_idx][n_neg] -= 1
        b[self.branch_idx] = self.voltage(t)



        

# ---------------------------------------------------------------------------
# CurrentSource
# ---------------------------------------------------------------------------

class CurrentSource(Element):
    """
    Ideal current source: injects I(t) from nodes[1] to nodes[0].
    This is the simplest stamp — it only touches b, not G.

        b[n+] -= I(t)
        b[n-] += I(t)
    """

    def __init__(self, name: str, nodes: list[int], I):
        super().__init__(name, nodes)
        self.I = I  # float or callable

    def current(self, t: float) -> float:
        if callable(self.I):
            return self.I(t)
        return float(self.I)

    def stamp(self, G, b, t, dt):
        # TODO: two lines — inject into b, skip ground nodes
        n_pos, n_neg = self.nodes[0], self.nodes[1]
        I_val = self.current(t)
        if n_pos != -1:
            b[n_pos] -= I_val
        if n_neg != -1:
            b[n_neg] += I_val