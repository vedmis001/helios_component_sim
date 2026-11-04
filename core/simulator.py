import numpy as np

from .circuit import Circuit


class Simulator:
    """
    Fixed-timestep time-domain driver.

    Results:
        self.time  shape (n_steps + 1,)
        self.x     shape (n_steps + 1, n_unknowns); row k is the full
                   solution vector at self.time[k]. Row 0 is t = 0.
    """

    def __init__(self, circuit: Circuit, dt: float, t_end: float):
        self.circuit = circuit
        self.dt = dt
        self.t_end = t_end
        self.n_steps = int(round(t_end / dt))

        self.time = np.zeros(self.n_steps + 1)
        self.x = np.zeros((self.n_steps + 1, circuit.n_unknowns))

    def run(self) -> None:
        """
        The time loop. Each step solves for the NEW time, t_new = (step + 1) * dt,
        using the histories left over from the previous step.

        Per step:
          1. t_new = (step + 1) * dt
          2. G, b = circuit.build_system(t_new, dt)
          3. x_new = circuit.solve(G, b)
          4. store t_new and x_new in self.time / self.x
          5. circuit.update_histories(x_new)

        Initial conditions: row 0 of self.x is all zeros, meaning every node
        voltage and every branch current starts at zero.
        """
        for step in range(1, self.n_steps + 1):
            t_new = step  * self.dt
            (G, b) = self.circuit.build_system(t_new, self.dt)
            x_new = self.circuit.solve(G, b)
            self.time[step] = t_new
            self.x[step] = x_new
            self.circuit.update_histories(x_new, self.dt)


        


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