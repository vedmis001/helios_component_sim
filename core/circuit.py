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
from .sparse import SparseMatrix


class SingularMatrixError(Exception):
    def __init__(self, message="Matrix is singular (non-invertible), check for floating nodes or missing gnd paths."):
        self.message = message
        super().__init__(self.message)

GROUND_NAMES = {"gnd", "0"}

ELEMENT_REGISTRY = {
    "resistor": Resistor,
    "capacitor": Capacitor,
    "inductor": Inductor,
    "vsource": VoltageSource,
    "isource": CurrentSource,
}

class Circuit:
    """
    Owns the elements and the node bookkeeping, and assembles G and b.

    Unknown vector layout (length n_unknowns):
        x[0 : n_nodes]                     node voltages
        x[n_nodes : n_nodes + n_vsources]  voltage-source branch currents
    """
    #node names that mean "reference node"
    # these map to index -1 and never as a row/column of the matrix.
    ground_names = GROUND_NAMES
    element_registry = ELEMENT_REGISTRY

    def __init__(self, node_map: dict, n_nodes: int, elements: list[Element], use_sparse: bool = True):
        self.node_map = node_map      # name -> matrix index (-1 for ground)
        self.n_nodes = n_nodes        # non-ground nodes only
        self.elements = elements
        self.use_sparse = use_sparse

        branch_idx = self.n_nodes

        # Voltage sources each add 1 branch unknown
        self.vsources = [e for e in elements if isinstance(e, VoltageSource)]
        for vs in self.vsources:
            vs.branch_idx = branch_idx
            branch_idx += 1

        # Multi-branch / component elements (Transformers, Battery, BuckBoost, Machine)
        self.branch_elements = [
            e for e in elements 
            if not isinstance(e, VoltageSource) and getattr(e, "n_branches", 0) > 0
        ]
        for el in self.branch_elements:
            el.branch_idx = branch_idx
            branch_idx += getattr(el, "n_branches", 1)

        self.n_unknowns = branch_idx

        # Pre-allocate vectorized COO triplet buffer (row, col, val) of size n_elements * 100
        self._sparse_G = SparseMatrix(
            n_unknowns=self.n_unknowns,
            n_elements=len(self.elements),
            capacity_per_element=100,
        )

    # ------------------------------------------------------------------
    # Construction from a config dict (plumbing, not the numerical core)
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict, use_sparse: bool = True) -> "Circuit":
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

        return cls(node_map, n_nodes, elements, use_sparse=use_sparse)

    # ------------------------------------------------------------------
    # Numerical core
    # ------------------------------------------------------------------
    def build_system(self, t: float, dt: float):
        """
        Assemble the MNA system for the solution at time t.

        Returns (G, b) where:
          - If `self.use_sparse` is True (default), G is a pre-allocated `SparseMatrix`
            triplet buffer (row, col, val vectors of capacity `n_elements * 100`)
          - If `self.use_sparse` is False, G is a dense `(n_unknowns x n_unknowns)` array
          - b is `(n_unknowns,)`
        """
        if self.use_sparse:
            self._sparse_G.reset()
            G = self._sparse_G
        else:
            G = np.zeros((self.n_unknowns, self.n_unknowns))
        b = np.zeros(self.n_unknowns)
        
        for elmt in self.elements:
            elmt.stamp(G, b, t, dt)

        return (G, b)
    
    def solve(self, G, b: np.ndarray) -> np.ndarray:
        """
        Solve G x = b and return x.
        Supports both `SparseMatrix` (vectorized COO -> CSC sparse LU) and dense `np.ndarray`.
        Raises `SingularMatrixError` on a singular matrix.
        """
        try:
            if isinstance(G, SparseMatrix):
                return G.solve(b)
            return np.linalg.solve(G, b)
        except np.linalg.LinAlgError as e:
            raise SingularMatrixError("Singular matrix encountered. Possible floating node.") from e

    def density_summary(self, t: float = 0.0, dt: float = 1e-6) -> dict:
        """Return matrix density/sparsity and triplet buffer capacity metrics."""
        G, _ = self.build_system(t, dt)
        if isinstance(G, SparseMatrix):
            return G.density_summary()
        nnz = int(np.count_nonzero(G))
        total = self.n_unknowns * self.n_unknowns
        density = float(nnz) / float(total) if total > 0 else 0.0
        return {
            "n_unknowns": self.n_unknowns,
            "total_entries": total,
            "stamped_triplets": nnz,
            "allocated_capacity": total,
            "nnz": nnz,
            "density": density,
            "sparsity": 1.0 - density,
        }

    def update_histories(self, x: np.ndarray, dt: float = 1e-6) -> None:
        """
        After a solve, let every dynamic element update its history.
        Passes dt if the component accepts it.
        """
        for element in self.elements:
            try:
                element.update_history(x, dt)
            except TypeError:
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
