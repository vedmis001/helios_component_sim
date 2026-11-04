"""
PSCAD Comparison & Validation Utility.

Supports:
1. Modern PSCAD v5+ binary output (.psout) via `mhi.psout`
2. Legacy PSCAD output (.inf / .out files)
3. Python simulation results (.csv)

Aligns traces, computes relative RMSE / Max Error, and plots overlays.
"""

import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mhi.psout
    HAS_MHI_PSOUT = True
except ImportError:
    HAS_MHI_PSOUT = False


def load_psout_data(psout_path: str) -> dict:
    """
    Loads traces from modern PSCAD .psout file using mhi.psout.
    Returns: {channel_name: numpy_array, "time": numpy_array}
    """
    if not HAS_MHI_PSOUT:
        raise ImportError(
            "mhi.psout is not installed. Install it with: pip install mhi.psout"
        )
    
    data = {}
    with mhi.psout.File(psout_path) as ps_file:
        runs = ps_file.runs() if callable(getattr(ps_file, "runs", None)) else ps_file.runs
        for run in runs:
            traces = run.traces() if hasattr(run, "traces") else getattr(run, "channels", [])
            for trace in traces:
                desc = (
                    trace.get("Description")
                    or trace.get("Component")
                    or trace.get("Name")
                    or (trace.call.get("Description") if getattr(trace, "call", None) else None)
                    or (trace.call.get("Name") if getattr(trace, "call", None) else None)
                    or getattr(trace, "name", None)
                    or getattr(trace, "title", None)
                    or str(trace.id)
                )
                arr = np.asarray(trace.data, dtype=float)
                if desc == "Domain" or trace.get("Group") == "Domain":
                    if "time" not in data:
                        data["time"] = arr
                    continue

                data[desc] = arr
                if "time" not in data:
                    if getattr(trace, "domain", None) is not None and getattr(trace.domain, "data", None) is not None:
                        data["time"] = np.asarray(trace.domain.data, dtype=float)
                    elif getattr(trace, "time", None) is not None:
                        data["time"] = np.asarray(trace.time, dtype=float)
    return data


def parse_legacy_inf(inf_path: str) -> dict:
    """Parses legacy PSCAD .inf file mapping."""
    channels = {}
    pattern = re.compile(r'PGB\((\d+)\)\s+Output\s+desc="([^"]+)"')
    with open(inf_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ch_idx = int(match.group(1))
                name = match.group(2).strip()
                file_idx = (ch_idx - 1) // 10 + 1
                col_idx = (ch_idx - 1) % 10 + 1
                channels[name] = (file_idx, col_idx)
    return channels


def load_legacy_pscad_data(case_prefix: str) -> dict:
    """Loads legacy PSCAD .out / .inf data."""
    inf_path = case_prefix + ".inf"
    channels = parse_legacy_inf(inf_path)
    data = {}
    
    needed_files = set(f_idx for f_idx, _ in channels.values())
    raw_files = {}
    for f_idx in needed_files:
        for out_file in [f"{case_prefix}_{f_idx:02d}.out", f"{case_prefix}_{f_idx}.out"]:
            if os.path.exists(out_file):
                raw_files[f_idx] = np.loadtxt(out_file)
                break

    if not raw_files:
        raise FileNotFoundError(f"Could not find .out files for prefix '{case_prefix}'")

    first_file = next(iter(raw_files.values()))
    data["time"] = first_file[:, 0]

    for name, (f_idx, col_idx) in channels.items():
        if f_idx in raw_files:
            data[name] = raw_files[f_idx][:, col_idx]

    return data


def load_pscad(path: str) -> dict:
    """Universal loader: auto-detects .psout vs legacy .inf/.out."""
    if path.endswith(".psout") or os.path.isfile(path + ".psout"):
        psout_file = path if path.endswith(".psout") else path + ".psout"
        return load_psout_data(psout_file)
    return load_legacy_pscad_data(path)


def load_sim_csv(csv_path: str) -> dict:
    """Loads Python simulation results saved via run.py / save_csv."""
    raw = np.genfromtxt(csv_path, delimiter=",", names=True)
    return {name: raw[name] for name in raw.dtype.names}


def compare_signals(t_ref, y_ref, t_sim, y_sim, label="Signal", plot_out=None):
    """Interpolates y_sim onto t_ref and calculates error metrics."""
    t_start = max(t_ref[0], t_sim[0])
    t_end = min(t_ref[-1], t_sim[-1])
    mask = (t_ref >= t_start) & (t_ref <= t_end)
    
    t_eval = t_ref[mask]
    y_ref_eval = y_ref[mask]
    y_sim_eval = np.interp(t_eval, t_sim, y_sim)

    err = y_sim_eval - y_ref_eval
    norm = np.max(np.abs(y_ref_eval)) if np.max(np.abs(y_ref_eval)) > 1e-6 else 1.0
    rmse = np.sqrt(np.mean(err**2)) / norm
    max_err = np.max(np.abs(err)) / norm

    print(f"[{label}] Relative RMSE = {rmse:.4e}, Relative Max Error = {max_err:.4e}")

    if plot_out:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        ax1.plot(t_eval * 1e3, y_ref_eval, label="PSCAD (Reference)", color="black", lw=1.5)
        ax1.plot(t_eval * 1e3, y_sim_eval, "--", label="HELIOS Python Simulator", color="crimson", lw=1.5)
        ax1.set_ylabel(label)
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_eval * 1e3, err, color="blue", lw=1.0)
        ax2.set_ylabel("Error (Diff)")
        ax2.set_xlabel("Time (ms)")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_out, dpi=150)
        print(f"Saved validation plot to {plot_out}")

    return rmse, max_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Python simulation against PSCAD traces.")
    parser.add_argument("--pscad", required=True, help="Path to .psout file or legacy prefix")
    parser.add_argument("--sim", required=True, help="Path to Python simulation CSV result")
    parser.add_argument("--signal", default="Vdc", help="Signal name to compare")
    parser.add_argument("--out", default="comparison.png", help="Path to save comparison plot")
    args = parser.parse_args()

    pscad = load_pscad(args.pscad)
    sim = load_sim_csv(args.sim)

    # Allow matching with or without prefix (e.g. 'v(n1)' or 'n1')
    sim_key = args.signal
    if sim_key not in sim:
        for k in sim:
            if args.signal in k:
                sim_key = k
                break

    pscad_key = args.signal
    if pscad_key not in pscad:
        for k in pscad:
            if args.signal in k:
                pscad_key = k
                break

    if pscad_key in pscad and sim_key in sim:
        compare_signals(pscad["time"], pscad[pscad_key], sim["time"], sim[sim_key],
                        label=args.signal, plot_out=args.out)
    else:
        print(f"Could not find matching signal '{args.signal}' in both datasets.")
        print(f"Available PSCAD signals: {list(pscad.keys())}")
        print(f"Available Sim signals:   {list(sim.keys())}")
