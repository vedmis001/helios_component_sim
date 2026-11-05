"""
Pre-allocated COO triplet (row, col, val) sparse matrix buffer for MNA stamping.

Allocates contiguous 1D NumPy vectors (rows, cols, vals) sized to
`max(n_elements * 100, 100)` so elements can stamp in O(1) without allocating
an (N x N) dense zero matrix at every time step.
"""

import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


class _RowProxy:
    """Allows chained indexing `G[row][col] += val` in addition to `G[row, col] += val`."""
    __slots__ = ("_mat", "_row")

    def __init__(self, mat: "SparseMatrix", row: int):
        self._mat = mat
        self._row = int(row)

    def __getitem__(self, col: int) -> float:
        return self._mat._get_entry(self._row, int(col))

    def __setitem__(self, col: int, val: float) -> None:
        self._mat._set_entry(self._row, int(col), float(val))


class SparseMatrix:
    """
    Vectorized COO (row, col, val) sparse matrix buffer with density tracking.

    Pre-allocates 1D NumPy arrays `rows`, `cols`, `vals` of length
    `capacity = max(n_elements * 100, 100)`.

    Supports:
    - O(1) scalar stamping via `G.add(r, c, v)` or `G[r, c] += v` or `G[r][c] += v`
    - Vectorized batch stamping via `G.add_batch(rows_arr, cols_arr, vals_arr)`
    - O(1) reset between time steps via `G.reset()` (reuses pre-allocated vectors)
    - Fast conversion to `scipy.sparse.csc_matrix` and density/sparsity metrics
    """

    def __init__(self, n_unknowns: int, n_elements: int = 1, capacity_per_element: int = 100,
                 dense_fallback_threshold: float = 0.65):
        self.n_unknowns = int(n_unknowns)
        self.shape = (self.n_unknowns, self.n_unknowns)
        self.n_elements = max(int(n_elements), 1)
        self.capacity = max(self.n_elements * int(capacity_per_element), 100)
        self.dense_fallback_threshold = float(dense_fallback_threshold)

        # Pre-allocated triplet vectors
        self.rows = np.empty(self.capacity, dtype=np.int32)
        self.cols = np.empty(self.capacity, dtype=np.int32)
        self.vals = np.empty(self.capacity, dtype=np.float64)
        self.ptr = 0

        # Cached coalesced CSC matrix (invalidated whenever new entries are stamped)
        self._csc_cache = None

    def reset(self) -> None:
        """Reset write pointer to 0 in O(1) without reallocating triplet arrays."""
        self.ptr = 0
        self._csc_cache = None

    def _ensure_capacity(self, additional: int) -> None:
        needed = self.ptr + additional
        if needed <= self.capacity:
            return
        new_cap = max(self.capacity * 2, needed + 100)
        new_rows = np.empty(new_cap, dtype=np.int32)
        new_cols = np.empty(new_cap, dtype=np.int32)
        new_vals = np.empty(new_cap, dtype=np.float64)
        new_rows[:self.ptr] = self.rows[:self.ptr]
        new_cols[:self.ptr] = self.cols[:self.ptr]
        new_vals[:self.ptr] = self.vals[:self.ptr]
        self.rows = new_rows
        self.cols = new_cols
        self.vals = new_vals
        self.capacity = new_cap

    def add(self, row: int, col: int, val: float) -> None:
        """Append a single (row, col, val) triplet in O(1)."""
        if row < 0 or col < 0 or val == 0.0:
            return
        self._ensure_capacity(1)
        p = self.ptr
        self.rows[p] = int(row)
        self.cols[p] = int(col)
        self.vals[p] = float(val)
        self.ptr = p + 1
        self._csc_cache = None

    def add_batch(self, rows, cols, vals) -> None:
        """Vectorized append of multiple (row, col, val) triplets."""
        r = np.asarray(rows, dtype=np.int32)
        c = np.asarray(cols, dtype=np.int32)
        v = np.asarray(vals, dtype=np.float64)
        mask = (r >= 0) & (c >= 0) & (v != 0.0)
        if not np.any(mask):
            return
        r = r[mask]
        c = c[mask]
        v = v[mask]
        k = len(v)
        self._ensure_capacity(k)
        p = self.ptr
        self.rows[p : p + k] = r
        self.cols[p : p + k] = c
        self.vals[p : p + k] = v
        self.ptr = p + k
        self._csc_cache = None

    def _get_entry(self, row: int, col: int) -> float:
        """Return current accumulated value at (row, col). Used by `G[r, c] += v`."""
        if self.ptr == 0:
            return 0.0
        r_slice = self.rows[:self.ptr]
        c_slice = self.cols[:self.ptr]
        mask = (r_slice == row) & (c_slice == col)
        if not np.any(mask):
            return 0.0
        return float(np.sum(self.vals[:self.ptr][mask]))

    def _set_entry(self, row: int, col: int, val: float) -> None:
        """
        Support `G[r, c] += delta` and `G[r, c] = val` from existing Element.stamp() methods.
        Computes the delta from the current accumulated value and appends a triplet,
        or appends directly if no prior entry exists at (row, col).
        """
        current = self._get_entry(row, col)
        delta = val - current
        if delta != 0.0:
            self.add(row, col, delta)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            if isinstance(row, slice) or isinstance(col, slice):
                return self.to_dense()[row, col]
            return self._get_entry(int(row), int(col))
        if isinstance(key, (int, np.integer)):
            return _RowProxy(self, int(key))
        return self.to_dense()[key]

    def __setitem__(self, key, val):
        if isinstance(key, tuple):
            row, col = key
            self._set_entry(int(row), int(col), float(val))
        else:
            raise TypeError("SparseMatrix assignment requires (row, col) index")

    def to_csc(self) -> sparse.csc_matrix:
        """Convert active triplet slice `[:ptr]` into a coalesced `scipy.sparse.csc_matrix`."""
        if self._csc_cache is None:
            if self.ptr == 0:
                self._csc_cache = sparse.csc_matrix(self.shape, dtype=np.float64)
            else:
                coo = sparse.coo_matrix(
                    (self.vals[:self.ptr], (self.rows[:self.ptr], self.cols[:self.ptr])),
                    shape=self.shape,
                    dtype=np.float64,
                )
                csc = coo.tocsc()
                csc.eliminate_zeros()
                self._csc_cache = csc
        return self._csc_cache

    def to_dense(self) -> np.ndarray:
        """Convert to a standard 2D NumPy array."""
        return self.to_csc().toarray()

    def __array__(self, dtype=None):
        arr = self.to_dense()
        return arr.astype(dtype) if dtype is not None else arr

    @property
    def nnz(self) -> int:
        """Number of structural non-zeros after coalescing duplicate triplets."""
        return int(self.to_csc().nnz)

    @property
    def density(self) -> float:
        """Fraction of non-zero entries: nnz / (n_unknowns^2)."""
        total = self.n_unknowns * self.n_unknowns
        if total == 0:
            return 0.0
        return float(self.nnz) / float(total)

    @property
    def sparsity(self) -> float:
        """Fraction of zero entries: 1.0 - density."""
        return 1.0 - self.density

    def density_summary(self) -> dict:
        """Return diagnostic dictionary of triplet buffer usage and matrix density."""
        return {
            "n_unknowns": self.n_unknowns,
            "total_entries": self.n_unknowns * self.n_unknowns,
            "stamped_triplets": int(self.ptr),
            "allocated_capacity": int(self.capacity),
            "nnz": self.nnz,
            "density": self.density,
            "sparsity": self.sparsity,
        }

    def solve(self, b: np.ndarray, force_sparse: bool = True) -> np.ndarray:
        """
        Solve G x = b using sparse LU (`spsolve`), keeping density in mind.
        Raises ` np.linalg.LinAlgError` on a singular / rank-deficient matrix so
        `Circuit.solve` can wrap it in `SingularMatrixError`.
        """
        if self.n_unknowns == 0:
            return np.zeros(0, dtype=np.float64)

        csc = self.to_csc()
        if csc.nnz == 0:
            raise np.linalg.LinAlgError("Singular matrix (all zeros)")

        # Check for completely empty rows or columns (fast structural singularity check)
        if np.any(np.diff(csc.indptr) == 0):
            raise np.linalg.LinAlgError("Singular matrix (empty column)")
        csr = csc.tocsr()
        if np.any(np.diff(csr.indptr) == 0):
            raise np.linalg.LinAlgError("Singular matrix (empty row)")

        # If force_sparse is False and matrix is tiny & very dense, dense LAPACK is faster;
        # otherwise use sparse SuperLU / UMFPACK via spsolve.
        if not force_sparse and self.n_unknowns <= 12 and self.density >= self.dense_fallback_threshold:
            return np.linalg.solve(csc.toarray(), b)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=splinalg.MatrixRankWarning)
            try:
                x = splinalg.spsolve(csc, b)
            except (splinalg.MatrixRankWarning, RuntimeError, ValueError) as exc:
                raise np.linalg.LinAlgError("Singular sparse matrix") from exc

        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise np.linalg.LinAlgError("Singular matrix produced non-finite solution")

        # Residual sanity check to catch near-singular floating subcircuits that SuperLU
        # solves with 1e16+ garbage instead of raising MatrixRankWarning
        res_norm = float(np.max(np.abs(csc @ x - b)))
        b_scale = max(float(np.max(np.abs(b))), 1.0)
        x_scale = float(np.max(np.abs(x)))
        if res_norm > 1e-5 * b_scale or x_scale > 1e13 * b_scale:
            # Verify rank via dense SVD/LU if small, or raise LinAlgError
            raise np.linalg.LinAlgError("Singular or ill-conditioned matrix detected")

        return x

