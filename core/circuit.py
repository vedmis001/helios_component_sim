#circuit class, matrix assembly

import numpy as np

from .elements import (
    Element,
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
)


class SingularMatrixError(Exception):
    def __init__(self, message="Matrix is singular (non-invertible), check for floating nodes or missing gnd paths."):
        self.message = message
        super().__init__(self.message)


class Circuit:
    """
    Owns the elements and the node bookkeeping, and assembles G and b.

    Unknown vector layout (length n_unknowns):
        x[0 : n_nodes]                     node voltages
        x[n_nodes : n_nodes + n_vsources]  voltage-source branch currents
    """
    #node names that mean "reference node"
    # these map to index -1 and never as a row/column of the matrix.
    ground_names = {"gnd", "0"}

    #maps the "type" string in a config dict to the element class.
    element_registry = {
        "resistor": Resistor,
        "capacitor": Capacitor,
        "inductor": Inductor,
        "vsource": VoltageSource,
        "isource": CurrentSource,
    }

    def __init__(self, node_map: dict, n_nodes: int, elements: list[Element]):
        self.node_map = node_map      # name -> matrix index (-1 for ground)
        self.n_nodes = n_nodes        # non-ground nodes only
        self.elements = elements

        # Voltage sources each add one unknown (their branch current).
        # Give each one its row/col index in the augmented matrix.
        self.vsources = [e for e in elements if isinstance(e, VoltageSource)]
        for k, vs in enumerate(self.vsources):
            vs.branch_idx = self.n_nodes + k

        self.n_unknowns = self.n_nodes + len(self.vsources)

    # ------------------------------------------------------------------
    # Construction from a config dict (plumbing, not the numerical core)
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "Circuit":
        """
        config = {
            "nodes": ["gnd", "n1", "n2"],
            "elements": [
                {"type": "vsource", "name": "Vs", "nodes": ["n1", "gnd"], "value": 1.0},
                ...
            ],
        }
        Element nodes are [positive, negative]; the first node is "+".
        """
        node_map = {}
        idx = 0
        for name in config["nodes"]:
            if name in cls.ground_names:
                node_map[name] = -1
            else:
                node_map[name] = idx
                idx += 1
        n_nodes = idx

        elements = []
        for spec in config["elements"]:
            element_cls = cls.element_registry[spec["type"]]
            node_idx = [node_map[n] for n in spec["nodes"]]
            elements.append(element_cls(spec["name"], node_idx, spec["value"]))

        return cls(node_map, n_nodes, elements)

    # ------------------------------------------------------------------
    # Numerical core (yours to implement)
    # ------------------------------------------------------------------
    def build_system(self, t: float, dt: float):
        """
        Assemble the MNA system for the solution at time t.

        Returns (G, b) where G is (n_unknowns x n_unknowns), b is (n_unknowns,).
        Steps:
          1. start from zeros
          2. every element stamps itself into G and b
          3. return both
        """
        #making G

        G = np.zeros((self.n_unknowns, self.n_unknowns))
        b = np.zeros(self.n_unknowns)
        
        for elmt in self.elements:
            #check syntax on this
            elmt.stamp(G, b, t, dt)

        return (G, b)
    
    def solve(self, G: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve G x = b and return x.
        numpy.linalg.solve raises LinAlgError on a singular matrix.
        A singular G almost always means a floating node (no DC path to
        ground) or a missing/duplicated stamp. Decide how you want to
        report that.
        """
        try:
            return np.linalg.solve(G, b)
        except np.linalg.LinAlgError as e:
            raise SingularMatrixError("Singular matrix encountered. Possible floating node.") from e




    def update_histories(self, x: np.ndarray) -> None:
        """
        After a solve, let every dynamic element update its history.
        If you change update_history() to also take dt (see elements.py),
        change the call here to match.
        """
        for element in self.elements:
            element.update_history(x)

    # ------------------------------------------------------------------
    # Helpers for reading results
    # ------------------------------------------------------------------
    def node_index(self, name: str) -> int:
        """Matrix index of a node (-1 means ground)."""
        return self.node_map[name]

    def vsource_index(self, name: str) -> int:
        """Index in x of a voltage source's branch current."""
        for vs in self.vsources:
            if vs.name == name:
                return vs.branch_idx
        raise KeyError(f"No voltage source named {name!r}")
    
