import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from PySDM import Formulae
from PySDM.physics import si

from Settings import Settings
from Simulation import Simulation

# =========================
# User config
# =========================
BASE_DIR = Path(__file__).resolve().parent
AER_FILE = BASE_DIR / "data" / "aerosol_spectrum.txt"
ENABLE_COLLISIONS = True
OUTPUT_SEEDING_CASE = True
RUN_CONTROL = True
RUN_SEEDING = False
REUSE_EXISTING_RESULTS_ONLY = False

DT = 0.1 * si.s
T_MAX = 4.70 * si.h
P0 = 1019.40 * si.hPa
T0 = 273.54 * si.K
RH0 = 0.985
PARCEL_VOLUME = 1.0 * si.m**3
W_MIN = 1e-4 * si.m / si.s
W_MAX = 0.075 * si.m / si.s
W_PERIOD = 170 * si.min
INITIAL_AEROSOL_KAPPA = 0.30
SEEDING_KAPPA = 0.8
SPECTRUM_INTERVAL_S = 60.0
PNG_DPI = 600
COMPARE_DIRNAME = "compare"
OUTPUT_BASE_DIR = BASE_DIR / "results"
OUTPUT_SUBDIR_COLLISIONS_ON = "collision_on"
OUTPUT_SUBDIR_COLLISIONS_OFF = "collision_off"

# ===== style: based on period, slightly larger for settle heatmap/grid =====
AX_LABEL_FS = 32
TICK_FS = 30
LEGEND_FS = 30
PANEL_FS = 36
TITLE_FS = 32
SPINE_LW = 3.0
LW_MAIN = 4.2
LW_LWC = 4.2
ALPHA_MAIN = 0.94
ALPHA_LWC = 0.84
LEGEND_LINE_LW = 5.0
CONTROL_EXTRA_LW = 0.55

SPECTRUM_AX_LABEL_FS = 34
SPECTRUM_TICK_FS = 24
SPECTRUM_TITLE_FS = 28
GRID_SPECTRUM_AX_LABEL_FS = 36
GRID_SPECTRUM_TICK_FS = 26
GRID_SPECTRUM_TITLE_FS = 30

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "Times New Roman"


def injection_rate_seeding(t):
    return 1 if (5 * si.min < t < 10 * si.min) else 0


def injection_rate_no_seeding(_):
    return 0


def _collision_mode_subdir(enable_collisions: bool) -> str:
    return OUTPUT_SUBDIR_COLLISIONS_ON if enable_collisions else OUTPUT_SUBDIR_COLLISIONS_OFF


def prepare_output_directories(base_dir: str, enable_collisions: bool):
    base_dir = os.path.abspath(base_dir)
    dir_on = os.path.join(base_dir, OUTPUT_SUBDIR_COLLISIONS_ON)
    dir_off = os.path.join(base_dir, OUTPUT_SUBDIR_COLLISIONS_OFF)
    os.makedirs(dir_on, exist_ok=True)
    os.makedirs(dir_off, exist_ok=True)
    run_dir = dir_on if enable_collisions else dir_off
    compare_dir = os.path.join(run_dir, COMPARE_DIRNAME)
    os.makedirs(compare_dir, exist_ok=True)
    return base_dir, run_dir, compare_dir


D1 = np.arange(0, 20 + 2, 2)
D2 = np.arange(20, 50 + 3, 3)
BASE_D_EDGES_UM = np.concatenate([D1, D2[1:]])
BASE_D_EDGES_UM = BASE_D_EDGES_UM[1:]

NO_UPPER_MIN_D_UM = 2.0
NO_UPPER_BIN_STEP_UM = 3.0


def _parse_numeric_line(line: str) -> np.ndarray:
    line = line.split("#", 1)[0].strip()
    if not line:
        return np.array([], dtype=float)
    toks = line.replace(",", " ").split()
    return np.array([float(t) for t in toks], dtype=float)


def read_aerosol_3rows_dNdlog10D(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    if len(lines) < 3:
        raise ValueError(f"Expect 3 numeric rows in {path}, got {len(lines)}")
    D_center_nm = _parse_numeric_line(lines[0])
    dNdlog10D_cm3 = _parse_numeric_line(lines[1])
    D_edge_nm = _parse_numeric_line(lines[2])
    N_bin_cm3 = dNdlog10D_cm3
    r_dry_m = 0.5 * D_center_nm * 1e-9
    return r_dry_m, N_bin_cm3, D_center_nm, D_edge_nm


def _sanitize_spectrum(spec):
    spec_plot = np.array(spec, dtype=float)
    spec_plot[~np.isfinite(spec_plot)] = np.nan
    spec_plot[spec_plot <= 0] = np.nan
    return spec_plot


def _series_or_nan(out, key, ref_key="summary_1min_time_s"):
    if key in out:
        return np.asarray(out[key], dtype=float)
    n = len(np.asarray(out.get(ref_key, []), dtype=float))
    return np.full(n, np.nan, dtype=float)


def _ensure_optional_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    optional_cols = {
        "parcel_volume_m3": np.nan,
        "N_mult_all_count": np.nan,
        "N_wet_count": np.nan,
        "N_drop_all_count_from_product": np.nan,
        "N_from_n_drop_all_times_volume_count": np.nan,
        "n_drop_all_cm3": np.nan,
        "n_drop_ge2_no_upper_cm3": np.nan,
        "LWC_manualspec_g_m3": np.nan,
        "r_v_manualspec_um": np.nan,
        "r_eff_manualspec_um": np.nan,
        "LWC_ge2_no_upper_manualspec_g_m3": np.nan,
        "r_v_ge2_no_upper_manualspec_um": np.nan,
        "r_eff_ge2_no_upper_manualspec_um": np.nan,
        "n_drop_ge2_no_upper_manual_cm3": np.nan,
        "LWC_ge2_no_upper_manualdirect_g_m3": np.nan,
        "r_v_ge2_no_upper_manualdirect_um": np.nan,
        "r_eff_ge2_no_upper_manualdirect_um": np.nan,
        "max_diameter_um": np.nan,
    }
    df = df.copy()
    for col, fill in optional_cols.items():
        if col not in df.columns:
            df[col] = fill
    if "time_s" in df.columns and "time_min" not in df.columns:
        df["time_min"] = pd.to_numeric(df["time_s"], errors="coerce") / 60.0
    if "n_drop_all_cm3" in df.columns and "parcel_volume_m3" in df.columns:
        cur = pd.to_numeric(df.get("N_from_n_drop_all_times_volume_count", np.nan), errors="coerce")
        derived = pd.to_numeric(df["n_drop_all_cm3"], errors="coerce") * 1e6 * pd.to_numeric(df["parcel_volume_m3"], errors="coerce")
        if isinstance(cur, pd.Series):
            cur_arr = cur.to_numpy(dtype=float)
        else:
            cur_arr = np.asarray(cur, dtype=float)
        der_arr = np.asarray(derived, dtype=float)
        if cur_arr.size == 1:
            cur_arr = np.full(len(df), np.nan)
        mask = ~np.isfinite(cur_arr)
        cur_arr[mask] = der_arr[mask]
        df["N_from_n_drop_all_times_volume_count"] = cur_arr
    return df


def _style_axes(ax, tick_fs=TICK_FS):
    ax.tick_params(direction="out", length=6.0, width=SPINE_LW, labelsize=tick_fs)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)


def _style_external_legend(legend):
    if legend is None:
        return
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
    for line in legend.get_lines():
        line.set_linewidth(LEGEND_LINE_LW)


def _add_panel_label(ax, text, loc="upper_left"):
    if not text:
        return
    x = 0.02
    y = 0.98 if loc == "upper_left" else 0.04
    va = "top" if loc == "upper_left" else "bottom"
    ax.text(x, y, text, transform=ax.transAxes, fontsize=PANEL_FS, fontweight="bold", ha="left", va=va)


def _positive_limits_for_log(vals, min_floor=1e-2):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return None
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    log_min = np.floor(np.log10(max(vmin, min_floor)))
    log_max = np.ceil(np.log10(max(vmax, min_floor)))
    lower = max(10 ** log_min, min_floor)
    upper = max(10 ** log_max, lower * 10)
    return lower, upper


def _collect_finite_values(results, ycol):
    vals = []
    for res in results:
        df = _ensure_optional_summary_columns(res["summary_1min"])
        if ycol not in df.columns:
            continue
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y)
        if np.any(mask):
            vals.append(y[mask])
    return np.concatenate(vals) if vals else np.array([], dtype=float)


def _apply_sensible_ylim(ax, results, ycol):
    vals = _collect_finite_values(results, ycol)
    if vals.size == 0:
        return
    ymin = float(np.nanmin(vals))
    ymax = float(np.nanmax(vals))
    if ycol in ["n_drop_cm3", "n_drop_all_cm3", "n_drop_ge2_no_upper_cm3", "n_drop_ge2_no_upper_manual_cm3", "N_all_cm3", "N_gt50_cm3"]:
        bottom_frac, top_frac, clamp_zero = 0.03, 0.08, True
    elif ycol in ["LWC_g_m3", "LWC_manualspec_g_m3", "LWC_ge2_no_upper_manualspec_g_m3", "LWC_ge2_no_upper_manualdirect_g_m3"]:
        bottom_frac, top_frac, clamp_zero = 0.02, 0.04, True
    else:
        bottom_frac, top_frac, clamp_zero = 0.06, 0.08, False
    yr = ymax - ymin
    if not np.isfinite(yr) or yr <= 0:
        yr = max(abs(ymax), 1.0) * 0.15
    lower = ymin - bottom_frac * yr
    upper = ymax + top_frac * yr
    if clamp_zero and ymin >= 0:
        lower = max(0.0, lower)
    if upper <= lower:
        upper = lower + max(0.1, abs(lower) * 0.2)
    ax.set_ylim(lower, upper)


def plot_timeseries(t, y, out_png=None, xlabel="Time (min)", ylabel="", color="#1f77b4", logy=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y_plot = np.asarray(y, dtype=float)
    if logy:
        y_plot = np.where(y_plot > 0, y_plot, np.nan)
    ax.plot(t, y_plot, linewidth=LW_MAIN, color=color)
    ax.set_xlabel(xlabel, fontsize=AX_LABEL_FS, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AX_LABEL_FS, fontweight="bold")
    if logy:
        ax.set_yscale("log")
        lims = _positive_limits_for_log(y_plot)
        if lims is not None:
            ax.set_ylim(*lims)
    _style_axes(ax)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_dual_timeseries(t, y1, y2, label1, label2, out_png=None, xlabel="Time (min)", ylabel=""):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(t, y1, linewidth=LW_MAIN, label=label1, color="#7C3AED")
    ax.plot(t, y2, linewidth=LW_MAIN, label=label2, color="#FF8A00")
    ax.set_xlabel(xlabel, fontsize=AX_LABEL_FS, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AX_LABEL_FS, fontweight="bold")
    legend = ax.legend(frameon=False, fontsize=LEGEND_FS)
    _style_external_legend(legend)
    _style_axes(ax)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def _mean_spectrum_curve(spectrum_n_cm3, D_edges_um):
    spec_raw = np.array(spectrum_n_cm3, dtype=float)
    spec_raw[~np.isfinite(spec_raw)] = np.nan
    mean_spec = np.nanmean(spec_raw, axis=0)
    D_edges_um = np.asarray(D_edges_um, dtype=float)
    return 0.5 * (D_edges_um[:-1] + D_edges_um[1:]), mean_spec


def _dynamic_diameter_ticks(max_edge_um: float):
    max_edge_um = float(max_edge_um)
    if max_edge_um <= 50:
        return [10, 20, 30, 40, 50]
    if max_edge_um <= 80:
        return [10, 20, 30, 40, 50, 60, 70, 80]
    step = 10.0 if max_edge_um <= 120 else 20.0
    ticks = np.arange(10.0, max_edge_um + 0.1, step)
    return [int(x) if abs(x - int(x)) < 1e-12 else float(x) for x in ticks]


def _dynamic_x_ticks(max_edge_um: float):
    max_edge_um = float(max_edge_um)
    if max_edge_um <= 50:
        return [0, 10, 20, 30, 40, 50]
    if max_edge_um <= 100:
        step = 10
    elif max_edge_um <= 180:
        step = 20
    else:
        step = 30
    ticks = list(np.arange(0, max_edge_um + 0.1, step))
    if ticks[-1] != int(max_edge_um):
        ticks.append(int(np.ceil(max_edge_um / step) * step))
    return ticks


def plot_spectrum_heatmap_from_arrays(spectrum_time_s, spectrum_n_cm3, D_edges_um, case_name):
    t_min = np.asarray(spectrum_time_s, dtype=float) / 60.0
    spec = np.asarray(spectrum_n_cm3, dtype=float)
    D_edges_um = np.asarray(D_edges_um, dtype=float)
    d_center_um = 0.5 * (D_edges_um[:-1] + D_edges_um[1:])
    spec_plot = _sanitize_spectrum(spec)
    valid = spec_plot[np.isfinite(spec_plot)]
    if valid.size == 0:
        return
    cmap = plt.get_cmap("rainbow")
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    fig.subplots_adjust(right=0.86)
    pcm = ax.pcolormesh(
        t_min, d_center_um, spec_plot.T, shading="auto", cmap=cmap,
        norm=LogNorm(vmin=np.nanmin(valid), vmax=np.nanmax(valid))
    )
    cax = fig.add_axes([0.88, 0.16, 0.025, 0.70])
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label("N (cm$^{-3}$)", fontsize=SPECTRUM_AX_LABEL_FS, fontweight="bold")
    cbar.ax.tick_params(labelsize=SPECTRUM_TICK_FS, width=SPINE_LW, length=6.0)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    cbar.outline.set_linewidth(SPINE_LW)
    ax.set_xlabel("Time(min)", fontsize=SPECTRUM_AX_LABEL_FS, fontweight="bold")
    ax.set_ylabel("D (μm)", fontsize=SPECTRUM_AX_LABEL_FS, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(max(1.0, D_edges_um[0]), D_edges_um[-1])
    yticks = _dynamic_diameter_ticks(D_edges_um[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(y)}" if abs(y - int(y)) < 1e-12 else f"{y:g}" for y in yticks],
                       fontsize=SPECTRUM_TICK_FS, fontweight="bold")
    _style_axes(ax, tick_fs=SPECTRUM_TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(f"{case_name}_droplet_spectrum_1min.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mean_spectrum_from_arrays(spectrum_n_cm3, D_edges_um, case_name, color="#1f77b4"):
    spec_raw = np.array(spectrum_n_cm3, dtype=float)
    spec_raw[~np.isfinite(spec_raw)] = np.nan
    mean_spec = np.nanmean(spec_raw, axis=0)
    D_edges_um = np.asarray(D_edges_um, dtype=float)
    d_um = 0.5 * (D_edges_um[:-1] + D_edges_um[1:])
    mean_spec_plot = np.where(mean_spec > 0, mean_spec, np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(d_um, mean_spec_plot, linewidth=LW_MAIN, color=color)
    ax.set_xlabel("D (μm)", fontsize=SPECTRUM_AX_LABEL_FS, fontweight="bold")
    ax.set_ylabel("N (cm$^{-3}$)", fontsize=SPECTRUM_AX_LABEL_FS, fontweight="bold")
    xmax = max(50.0, float(D_edges_um[-1]))
    ax.set_xlim(0, xmax)
    ax.set_xticks(_dynamic_x_ticks(xmax))
    ax.set_yscale("log")
    _style_axes(ax, tick_fs=SPECTRUM_TICK_FS)
    fig.tight_layout()
    fig.savefig(f"{case_name}_mean_spectrum_1minavg.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_compare_2x2(results, out_png, n_logy=True,
                    ycols=None, ylabels=None, panel_labels=None):
    if ycols is None:
        ycols = ["n_drop_all_cm3", "LWC_g_m3", "Rv_um", "Reff_um"]
    if ylabels is None:
        ylabels = ["N (cm$^{-3}$)", "LWC (g m$^{-3}$)", "R$_v$ (µm)", "R$_{eff}$ (µm)"]
    if panel_labels is None:
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.8))
    axes = axes.ravel()
    for i, (ax, ycol, ylabel) in enumerate(zip(axes, ycols, ylabels)):
        for res in results:
            df = _ensure_optional_summary_columns(res["summary_1min"])
            if ycol not in df.columns:
                continue
            y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
            if i == 0 and n_logy:
                y = np.where(y > 0, y, np.nan)
            lw = LW_LWC if ycol in ["LWC_g_m3", "LWC_manualspec_g_m3", "LWC_ge2_no_upper_manualspec_g_m3", "LWC_ge2_no_upper_manualdirect_g_m3"] else LW_MAIN
            ax.plot(df["time_min"].to_numpy(dtype=float), y, linewidth=lw, color=res["color"], label=res["label"])
        ax.set_xlabel("Time (min)", fontsize=AX_LABEL_FS, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=AX_LABEL_FS, fontweight="bold")
        if i == 0 and n_logy:
            ax.set_yscale("log")
            lims = _positive_limits_for_log(_collect_finite_values(results, ycol), min_floor=1e-1)
            if lims is not None:
                ax.set_ylim(*lims)
        else:
            _apply_sensible_ylim(ax, results, ycol)
        _style_axes(ax)
        _add_panel_label(ax, panel_labels[i], loc="upper_left")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                        columnspacing=1.4, handlelength=2.5, fontsize=LEGEND_FS)
    _style_external_legend(legend)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.86, wspace=0.24, hspace=0.25)
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)




def plot_single_case_2x2(df_1min, out_png, ycols, ylabels, panel_labels=None, n_logy=False, line_label=None, line_color="#1f77b4"):
    df_1min = _ensure_optional_summary_columns(df_1min)
    if panel_labels is None:
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.8))
    axes = axes.ravel()
    pseudo_results = [{"summary_1min": df_1min, "label": line_label or "series", "color": line_color}]
    x = pd.to_numeric(df_1min["time_min"], errors="coerce").to_numpy(dtype=float)
    for i, (ax, ycol, ylabel) in enumerate(zip(axes, ycols, ylabels)):
        y = pd.to_numeric(df_1min[ycol], errors="coerce").to_numpy(dtype=float) if ycol in df_1min.columns else np.full_like(x, np.nan, dtype=float)
        if i == 0 and n_logy:
            y = np.where(y > 0, y, np.nan)
        lw = LW_LWC if ycol in ["LWC_g_m3", "LWC_manualspec_g_m3", "LWC_ge2_no_upper_manualspec_g_m3", "LWC_ge2_no_upper_manualdirect_g_m3"] else LW_MAIN
        ax.plot(x, y, linewidth=lw, color=line_color, label=line_label)
        ax.set_xlabel("Time (min)", fontsize=AX_LABEL_FS, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=AX_LABEL_FS, fontweight="bold")
        if i == 0 and n_logy:
            ax.set_yscale("log")
            lims = _positive_limits_for_log(y, min_floor=1e-1)
            if lims is not None:
                ax.set_ylim(*lims)
        else:
            _apply_sensible_ylim(ax, pseudo_results, ycol)
        _style_axes(ax)
        _add_panel_label(ax, panel_labels[i], loc="upper_left")
    if line_label:
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, frameon=False, ncol=1, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                            columnspacing=1.4, handlelength=2.5, fontsize=LEGEND_FS)
        _style_external_legend(legend)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.86, wspace=0.24, hspace=0.25)
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_three_method_compare_2x2(df_1min, out_png, method_specs, panel_labels=None):
    df_1min = _ensure_optional_summary_columns(df_1min)
    if panel_labels is None:
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    fields = ["N", "LWC", "Rv", "Reff"]
    ylabels = ["N (cm$^{-3}$)", "LWC (g m$^{-3}$)", "R$_v$ (µm)", "R$_{eff}$ (µm)"]
    x = pd.to_numeric(df_1min["time_min"], errors="coerce").to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.8))
    axes = axes.ravel()
    for i, (ax, field, ylabel) in enumerate(zip(axes, fields, ylabels)):
        vals_for_ylim = []
        for spec in method_specs:
            col = spec["cols"][field]
            y = pd.to_numeric(df_1min[col], errors="coerce").to_numpy(dtype=float) if col in df_1min.columns else np.full_like(x, np.nan, dtype=float)
            vals_for_ylim.append(y)
            lw = LW_LWC if field == "LWC" else LW_MAIN
            ax.plot(x, y, linewidth=lw, color=spec["color"], linestyle=spec.get("linestyle", "-"), label=spec["label"])
        ax.set_xlabel("Time (min)", fontsize=AX_LABEL_FS, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=AX_LABEL_FS, fontweight="bold")
        vals = np.concatenate([v[np.isfinite(v)] for v in vals_for_ylim if np.any(np.isfinite(v))]) if any(np.any(np.isfinite(v)) for v in vals_for_ylim) else np.array([], dtype=float)
        if vals.size:
            ymin = float(np.nanmin(vals))
            ymax = float(np.nanmax(vals))
            if field in ["N", "LWC"]:
                bottom_frac, top_frac, clamp_zero = (0.03, 0.08, True) if field == "N" else (0.02, 0.04, True)
            else:
                bottom_frac, top_frac, clamp_zero = 0.06, 0.08, False
            yr = ymax - ymin
            if not np.isfinite(yr) or yr <= 0:
                yr = max(abs(ymax), 1.0) * 0.15
            lower = ymin - bottom_frac * yr
            upper = ymax + top_frac * yr
            if clamp_zero and ymin >= 0:
                lower = max(0.0, lower)
            if upper <= lower:
                upper = lower + max(0.1, abs(lower) * 0.2)
            ax.set_ylim(lower, upper)
        _style_axes(ax)
        _add_panel_label(ax, panel_labels[i], loc="upper_left")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                        columnspacing=1.2, handlelength=2.7, fontsize=max(LEGEND_FS - 1, 10))
    _style_external_legend(legend)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.86, wspace=0.24, hspace=0.25)
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)

def _spectrum_color_limits(results, spectrum_key="spectrum"):
    all_valid = []
    for res in results:
        spec = _sanitize_spectrum(res[spectrum_key]["spectrum_n_cm3"])
        valid = spec[np.isfinite(spec)]
        if valid.size:
            all_valid.append(valid)
    if not all_valid:
        return 1e-3, 1e1
    return float(min(np.nanmin(v) for v in all_valid)), float(max(np.nanmax(v) for v in all_valid))


def plot_compare_mean_spectrum(results, out_png, spectrum_key="spectrum"):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    xmax = 50.0
    has_curve = False
    for res in results:
        spec = np.asarray(res[spectrum_key]["spectrum_n_cm3"], dtype=float)
        edges = np.asarray(res[spectrum_key]["D_edges_um"], dtype=float)
        d_um, mean_spec = _mean_spectrum_curve(spec, edges)
        mean_spec = np.where(mean_spec > 0, mean_spec, np.nan)
        if np.all(~np.isfinite(mean_spec)):
            continue
        ax.plot(d_um, mean_spec, linewidth=LW_MAIN, color=res["color"], label=res["label"])
        xmax = max(xmax, float(edges[-1]))
        has_curve = True
    if not has_curve:
        plt.close(fig)
        return
    ax.set_xlabel("D (μm)", fontsize=SPECTRUM_AX_LABEL_FS, fontweight="bold")
    ax.set_ylabel("N (cm$^{-3}$)", fontsize=SPECTRUM_AX_LABEL_FS, fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_xticks(_dynamic_x_ticks(xmax))
    ax.set_yscale("log")
    legend = ax.legend(frameon=False, ncol=2, fontsize=LEGEND_FS)
    _style_external_legend(legend)
    _style_axes(ax, tick_fs=SPECTRUM_TICK_FS)
    fig.tight_layout()
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_grid(results, out_png, spectrum_key="spectrum"):
    n = len(results)
    ncols = 2 if n <= 2 else 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 6.1 * nrows), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.83, bottom=0.10, top=0.93, wspace=0.54, hspace=0.56)
    cmap = plt.get_cmap("rainbow")
    vmin, vmax = _spectrum_color_limits(results, spectrum_key=spectrum_key)
    pcm = None
    for i, res in enumerate(results):
        ax = axes.ravel()[i]
        t_min = np.asarray(res[spectrum_key]["spectrum_time_s"], dtype=float) / 60.0
        spec = _sanitize_spectrum(res[spectrum_key]["spectrum_n_cm3"])
        edges = np.asarray(res[spectrum_key]["D_edges_um"], dtype=float)
        d_center_um = 0.5 * (edges[:-1] + edges[1:])
        pcm = ax.pcolormesh(t_min, d_center_um, spec.T, shading="auto", cmap=cmap,
                            norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xticks([0, 50, 100, 150])
        ax.set_xlabel("Time(min)", fontsize=GRID_SPECTRUM_AX_LABEL_FS, fontweight="bold", labelpad=8)
        ax.set_ylabel("D (μm)", fontsize=GRID_SPECTRUM_AX_LABEL_FS, fontweight="bold", labelpad=8)
        ax.set_yscale("log")
        ax.set_ylim(max(1.0, edges[0]), edges[-1])
        yticks = _dynamic_diameter_ticks(edges[-1])
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(y)}" if abs(y - int(y)) < 1e-12 else f"{y:g}" for y in yticks],
                           fontsize=GRID_SPECTRUM_TICK_FS, fontweight="bold")
        _style_axes(ax, tick_fs=GRID_SPECTRUM_TICK_FS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(res["title"], fontsize=GRID_SPECTRUM_TITLE_FS, fontweight="bold", pad=12)
        ax.text(0.02, 0.98, f"({chr(97+i)})", transform=ax.transAxes,
                fontsize=PANEL_FS - 6, fontweight="bold", ha="left", va="top")
    for j in range(n, nrows * ncols):
        axes.ravel()[j].axis("off")
    cax = fig.add_axes([0.88, 0.18, 0.022, 0.64])
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label("N (cm$^{-3}$)", fontsize=GRID_SPECTRUM_AX_LABEL_FS, fontweight="bold")
    cbar.ax.tick_params(labelsize=GRID_SPECTRUM_TICK_FS + 2, width=SPINE_LW + 0.6, length=6.0)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    cbar.outline.set_linewidth(SPINE_LW + 0.6)
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def dump_run_config_txt(case_name: str, settings, out_txt: str = None):
    if out_txt is None:
        out_txt = f"{case_name}_run_config.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"CASE = {case_name}\n")
        f.write("=" * 40 + "\n")
        f.write(f"P0 = {float(settings.initial_total_pressure)/100:.2f} hPa\n")
        f.write(f"T0 = {float(settings.initial_temperature):.2f} K\n")
        f.write(f"RH0 = {getattr(settings, 'initial_RH', np.nan)}\n")
        f.write(f"W_MIN = {float(settings.w_min):.6g} m/s\n")
        f.write(f"W_MAX = {float(settings.w_max):.6g} m/s\n")
        f.write(f"W_PERIOD = {float(settings.updraft_period)/60:.2f} min\n")
        f.write(f"initial_aerosol_kappa = {getattr(settings, 'initial_aerosol_kappa', np.nan)}\n")
        f.write(f"seeding_kappa = {getattr(settings, 'seeding_kappa', np.nan)}\n")
        f.write(f"ENABLE_COLLISIONS = {settings.enable_collisions}\n")
        f.write(f"OUTPUT_SEEDING_CASE = {settings.output_seeding_case}\n")


def save_summary_csv(out, prefix):
    max_diameter_um = float(out.get("max_diameter_um", np.nan))

    df_dt = pd.DataFrame({
        "time_s": out["time_s"],
        "time_min": np.asarray(out["time_s"], dtype=float) / 60.0,
        "n_drop_cm3": out["n_drop_cm3"],
        "n_drop_ge2_no_upper_cm3": out["n_drop_ge2_no_upper_cm3"],
        "n_drop_all_cm3": out["n_drop_all_cm3"],
        "LWC_g_m3": out["LWC_g_m3"],
        "Rv_um": out["r_v_um"],
        "Reff_um": out["r_eff_um"],
        "supersaturation_pct": out["supersaturation_pct"],
        "N_all_cm3": out["N_all_cm3"],
        "N_gt50_cm3": out["N_gt50_cm3"],
        "N_lt1um_cm3": out["N_lt1um_cm3"],
        "N_1to25um_manual_cm3": out["N_1to25um_manual_cm3"],
        "N_gt25um_radius_cm3": out["N_gt25um_radius_cm3"],
        "parcel_volume_m3": _series_or_nan(out, "parcel_volume_m3", ref_key="time_s"),
        "N_mult_all_count": _series_or_nan(out, "N_mult_all_count", ref_key="time_s"),
        "N_wet_count": _series_or_nan(out, "N_wet_count", ref_key="time_s"),
        "N_drop_all_count_from_product": _series_or_nan(out, "N_drop_all_count_from_product", ref_key="time_s"),
        "LWC_manualspec_g_m3": out["LWC_manualspec_g_m3"],
        "r_v_manualspec_um": out["r_v_manualspec_um"],
        "r_eff_manualspec_um": out["r_eff_manualspec_um"],
        "LWC_ge2_no_upper_manualspec_g_m3": out["LWC_ge2_no_upper_manualspec_g_m3"],
        "r_v_ge2_no_upper_manualspec_um": out["r_v_ge2_no_upper_manualspec_um"],
        "r_eff_ge2_no_upper_manualspec_um": out["r_eff_ge2_no_upper_manualspec_um"],
        "n_drop_ge2_no_upper_manual_cm3": out["n_drop_ge2_no_upper_manual_cm3"],
        "LWC_ge2_no_upper_manualdirect_g_m3": out["LWC_ge2_no_upper_manualdirect_g_m3"],
        "r_v_ge2_no_upper_manualdirect_um": out["r_v_ge2_no_upper_manualdirect_um"],
        "r_eff_ge2_no_upper_manualdirect_um": out["r_eff_ge2_no_upper_manualdirect_um"],
        "max_diameter_um": np.full(len(out["time_s"]), max_diameter_um, dtype=float),
    })
    df_dt = _ensure_optional_summary_columns(df_dt)
    df_dt.to_csv(f"{prefix}_summary_timeseries_dt.csv", index=False)

    df_1min = pd.DataFrame({
        "time_s": out["summary_1min_time_s"],
        "time_min": np.asarray(out["summary_1min_time_s"], dtype=float) / 60.0,
        "n_drop_cm3": out["summary_1min_n_drop_cm3"],
        "n_drop_ge2_no_upper_cm3": out["summary_1min_n_drop_ge2_no_upper_cm3"],
        "n_drop_all_cm3": out["summary_1min_n_drop_all_cm3"],
        "LWC_g_m3": out["summary_1min_LWC_g_m3"],
        "Rv_um": out["summary_1min_r_v_um"],
        "Reff_um": out["summary_1min_r_eff_um"],
        "supersaturation_pct": out["summary_1min_supersaturation_pct"],
        "N_all_cm3": out["summary_1min_N_all_cm3"],
        "N_gt50_cm3": out["summary_1min_N_gt50_cm3"],
        "N_lt1um_cm3": out["summary_1min_N_lt1um_cm3"],
        "N_1to25um_manual_cm3": out["summary_1min_N_1to25um_manual_cm3"],
        "N_gt25um_radius_cm3": out["summary_1min_N_gt25um_radius_cm3"],
        "parcel_volume_m3": _series_or_nan(out, "summary_1min_parcel_volume_m3"),
        "N_mult_all_count": _series_or_nan(out, "summary_1min_N_mult_all_count"),
        "N_wet_count": _series_or_nan(out, "summary_1min_N_wet_count"),
        "N_drop_all_count_from_product": _series_or_nan(out, "summary_1min_N_drop_all_count_from_product"),
        "LWC_manualspec_g_m3": out["summary_1min_LWC_manualspec_g_m3"],
        "r_v_manualspec_um": out["summary_1min_r_v_manualspec_um"],
        "r_eff_manualspec_um": out["summary_1min_r_eff_manualspec_um"],
        "LWC_ge2_no_upper_manualspec_g_m3": out["summary_1min_LWC_ge2_no_upper_manualspec_g_m3"],
        "r_v_ge2_no_upper_manualspec_um": out["summary_1min_r_v_ge2_no_upper_manualspec_um"],
        "r_eff_ge2_no_upper_manualspec_um": out["summary_1min_r_eff_ge2_no_upper_manualspec_um"],
        "n_drop_ge2_no_upper_manual_cm3": out["summary_1min_n_drop_ge2_no_upper_manual_cm3"],
        "LWC_ge2_no_upper_manualdirect_g_m3": out["summary_1min_LWC_ge2_no_upper_manualdirect_g_m3"],
        "r_v_ge2_no_upper_manualdirect_um": out["summary_1min_r_v_ge2_no_upper_manualdirect_um"],
        "r_eff_ge2_no_upper_manualdirect_um": out["summary_1min_r_eff_ge2_no_upper_manualdirect_um"],
        "max_diameter_um": np.full(len(out["summary_1min_time_s"]), max_diameter_um, dtype=float),
    })
    df_1min = _ensure_optional_summary_columns(df_1min)
    df_1min.to_csv(f"{prefix}_summary_timeseries_1min.csv", index=False)

    df_manual = df_1min[[
        "time_s", "time_min", "n_drop_cm3", "LWC_manualspec_g_m3",
        "r_v_manualspec_um", "r_eff_manualspec_um"
    ]].copy()
    df_manual.rename(columns={
        "n_drop_cm3": "N_2to50um_cm3_from_product",
        "LWC_manualspec_g_m3": "LWC_2to50um_manualspec_g_m3",
        "r_v_manualspec_um": "Rv_2to50um_manualspec_um",
        "r_eff_manualspec_um": "Reff_2to50um_manualspec_um",
    }, inplace=True)
    df_manual.to_csv(f"{prefix}_manualbulk_2to50um_1min.csv", index=False)

    df_ge2 = df_1min[[
        "time_s", "time_min", "n_drop_ge2_no_upper_cm3",
        "LWC_ge2_no_upper_manualspec_g_m3",
        "r_v_ge2_no_upper_manualspec_um",
        "r_eff_ge2_no_upper_manualspec_um",
        "max_diameter_um",
    ]].copy()
    df_ge2.rename(columns={
        "n_drop_ge2_no_upper_cm3": "N_Dge2um_no_upper_cm3_from_product",
        "LWC_ge2_no_upper_manualspec_g_m3": "LWC_Dge2um_no_upper_manualspec_g_m3",
        "r_v_ge2_no_upper_manualspec_um": "Rv_Dge2um_no_upper_manualspec_um",
        "r_eff_ge2_no_upper_manualspec_um": "Reff_Dge2um_no_upper_manualspec_um",
    }, inplace=True)
    df_ge2.to_csv(f"{prefix}_manualbulk_Dge2um_no_upper_1min.csv", index=False)

    df_ge2_manualdirect = df_1min[[
        "time_s", "time_min", "n_drop_ge2_no_upper_manual_cm3",
        "LWC_ge2_no_upper_manualdirect_g_m3",
        "r_v_ge2_no_upper_manualdirect_um",
        "r_eff_ge2_no_upper_manualdirect_um",
        "max_diameter_um",
    ]].copy()
    df_ge2_manualdirect.rename(columns={
        "n_drop_ge2_no_upper_manual_cm3": "N_Dge2um_no_upper_cm3_manual_direct",
        "LWC_ge2_no_upper_manualdirect_g_m3": "LWC_Dge2um_no_upper_manualdirect_g_m3",
        "r_v_ge2_no_upper_manualdirect_um": "Rv_Dge2um_no_upper_manualdirect_um",
        "r_eff_ge2_no_upper_manualdirect_um": "Reff_Dge2um_no_upper_manualdirect_um",
    }, inplace=True)
    df_ge2_manualdirect.to_csv(f"{prefix}_manualbulk_Dge2um_no_upper_manualN_1min.csv", index=False)

    df_sup = df_1min[["time_s", "time_min", "supersaturation_pct"]].copy()
    df_sup.to_csv(f"{prefix}_supersaturation_1min.csv", index=False)

    df_num = df_1min[["time_s", "time_min", "N_all_cm3", "N_gt50_cm3"]].copy()
    df_num.to_csv(f"{prefix}_Nall_Ngt50_1min.csv", index=False)

    return df_dt, df_1min, df_sup, df_num


def save_spectrum_csv(spectrum_time_s, spectrum_n_cm3, D_edges_um, out_csv):
    edges = np.asarray(D_edges_um, dtype=float)
    left = edges[:-1]
    right = edges[1:]
    bin_cols = [f"bin_{l:g}_{r:g}" for l, r in zip(left, right)]
    times = np.asarray(spectrum_time_s, dtype=float)
    spec = np.asarray(spectrum_n_cm3, dtype=float)
    df = pd.DataFrame(spec, columns=bin_cols)
    df.insert(0, "time_s", times)
    df.to_csv(out_csv, index=False)


def parse_spectrum_edges_from_columns(columns):
    bin_cols = [c for c in columns if c != "time_s"]
    if not bin_cols:
        raise ValueError("No spectrum bin columns found.")
    lefts = []
    rights = []
    for col in bin_cols:
        if not col.startswith("bin_"):
            raise ValueError(f"Unexpected spectrum column name: {col}")
        _, left, right = col.split("_", 2)
        lefts.append(float(left))
        rights.append(float(right))
    edges = np.array([lefts[0], *rights], dtype=float)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Parsed spectrum edges are not strictly increasing.")
    return edges


def load_saved_case_outputs(prefix):
    f_1min = f"{prefix}_summary_timeseries_1min.csv"
    f_spec = f"{prefix}_droplet_spectrum.csv"
    f_spec_ge2 = f"{prefix}_Dge2um_no_upper_droplet_spectrum.csv"
    for fp in [f_1min, f_spec, f_spec_ge2]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Required file not found: {fp}")
    df_1min = _ensure_optional_summary_columns(pd.read_csv(f_1min))
    df_sup = df_1min[["time_s", "time_min", "supersaturation_pct"]].copy()
    df_num = df_1min[["time_s", "time_min", "N_all_cm3", "N_gt50_cm3"]].copy()

    df_spec = pd.read_csv(f_spec)
    spectrum_time_s = df_spec["time_s"].to_numpy(dtype=float)
    spectrum_n_cm3 = df_spec.drop(columns=["time_s"]).to_numpy(dtype=float)
    spectrum_edges_um = parse_spectrum_edges_from_columns(df_spec.columns)

    df_spec_ge2 = pd.read_csv(f_spec_ge2)
    spectrum_ge2_no_upper_time_s = df_spec_ge2["time_s"].to_numpy(dtype=float)
    spectrum_ge2_no_upper_n_cm3 = df_spec_ge2.drop(columns=["time_s"]).to_numpy(dtype=float)
    spectrum_ge2_no_upper_edges_um = parse_spectrum_edges_from_columns(df_spec_ge2.columns)

    return (
        df_1min, df_sup, df_num,
        spectrum_time_s, spectrum_n_cm3, spectrum_edges_um,
        spectrum_ge2_no_upper_time_s, spectrum_ge2_no_upper_n_cm3, spectrum_ge2_no_upper_edges_um,
    )


def make_all_plots(
    case_name,
    df_1min,
    df_sup,
    df_num,
    spectrum_time_s,
    spectrum_n_cm3,
    spectrum_edges_um,
    spectrum_ge2_no_upper_time_s,
    spectrum_ge2_no_upper_n_cm3,
    spectrum_ge2_no_upper_edges_um,
    color="#1f77b4",
):
    df_1min = _ensure_optional_summary_columns(df_1min)
    plot_timeseries(df_1min["time_min"].to_numpy(), df_1min["LWC_g_m3"].to_numpy(),
                    ylabel="LWC (g m$^{-3}$)", out_png=f"{case_name}_LWC_1min.png", color=color)
    plot_timeseries(df_1min["time_min"].to_numpy(), df_1min["n_drop_all_cm3"].to_numpy(),
                    ylabel="N (cm$^{-3}$)", out_png=f"{case_name}_N_1min.png", color=color)
    plot_timeseries(df_1min["time_min"].to_numpy(), df_1min["Reff_um"].to_numpy(),
                    ylabel="R$_{eff}$ (µm)", out_png=f"{case_name}_Reff_1min.png", color=color)
    plot_timeseries(df_1min["time_min"].to_numpy(), df_1min["Rv_um"].to_numpy(),
                    ylabel="R$_v$ (µm)", out_png=f"{case_name}_Rv_1min.png", color=color)
    plot_timeseries(df_sup["time_min"].to_numpy(), df_sup["supersaturation_pct"].to_numpy(),
                    ylabel="Supersaturation (%)", out_png=f"{case_name}_supersaturation_1min.png", color=color)
    plot_dual_timeseries(df_num["time_min"].to_numpy(), df_num["N_all_cm3"].to_numpy(), df_num["N_gt50_cm3"].to_numpy(),
                         label1="N_all", label2="N_gt50", ylabel="Number concentration (cm$^{-3}$)",
                         out_png=f"{case_name}_Nall_Ngt50_1min.png")
    plot_timeseries(df_1min["time_min"].to_numpy(), df_1min["LWC_manualspec_g_m3"].to_numpy(),
                    ylabel="LWC (g m$^{-3}$)", out_png=f"{case_name}_LWC_manualspec_1min.png", color=color)
    plot_timeseries(df_1min["time_min"].to_numpy(), df_1min["r_v_manualspec_um"].to_numpy(),
                    ylabel="R$_v$ (µm)", out_png=f"{case_name}_Rv_manualspec_1min.png", color=color)
    plot_timeseries(df_1min["time_min"].to_numpy(), df_1min["r_eff_manualspec_um"].to_numpy(),
                    ylabel="R$_{eff}$ (µm)", out_png=f"{case_name}_Reff_manualspec_1min.png", color=color)

    plot_single_case_2x2(
        df_1min,
        f"{case_name}_compare_2x2_2to50um_1min.png",
        ycols=["n_drop_cm3", "LWC_manualspec_g_m3", "r_v_manualspec_um", "r_eff_manualspec_um"],
        ylabels=["N (cm$^{-3}$)", "LWC (g m$^{-3}$)", "R$_v$ (µm)", "R$_{eff}$ (µm)"],
        n_logy=False,
        line_label="2–50 μm (product N)",
        line_color=color,
    )

    plot_spectrum_heatmap_from_arrays(spectrum_time_s, spectrum_n_cm3, spectrum_edges_um, case_name)
    plot_mean_spectrum_from_arrays(spectrum_n_cm3, spectrum_edges_um, case_name, color=color)

    no_upper_prefix = f"{case_name}_Dge2um_no_upper"
    plot_single_case_2x2(
        df_1min,
        f"{no_upper_prefix}_compare_2x2_1min.png",
        ycols=[
            "n_drop_ge2_no_upper_cm3",
            "LWC_ge2_no_upper_manualspec_g_m3",
            "r_v_ge2_no_upper_manualspec_um",
            "r_eff_ge2_no_upper_manualspec_um",
        ],
        ylabels=["N (cm$^{-3}$)", "LWC (g m$^{-3}$)", "R$_v$ (µm)", "R$_{eff}$ (µm)"],
        n_logy=False,
        line_label="D≥2 μm, no upper (product N)",
        line_color=color,
    )
    plot_spectrum_heatmap_from_arrays(
        spectrum_ge2_no_upper_time_s,
        spectrum_ge2_no_upper_n_cm3,
        spectrum_ge2_no_upper_edges_um,
        no_upper_prefix,
    )
    plot_mean_spectrum_from_arrays(
        spectrum_ge2_no_upper_n_cm3,
        spectrum_ge2_no_upper_edges_um,
        no_upper_prefix,
        color=color,
    )

    manualdirect_prefix = f"{case_name}_Dge2um_no_upper_manualN"
    plot_single_case_2x2(
        df_1min,
        f"{manualdirect_prefix}_compare_2x2_1min.png",
        ycols=[
            "n_drop_ge2_no_upper_manual_cm3",
            "LWC_ge2_no_upper_manualdirect_g_m3",
            "r_v_ge2_no_upper_manualdirect_um",
            "r_eff_ge2_no_upper_manualdirect_um",
        ],
        ylabels=["N (cm$^{-3}$)", "LWC (g m$^{-3}$)", "R$_v$ (µm)", "R$_{eff}$ (µm)"],
        n_logy=False,
        line_label="D≥2 μm, no upper (manual N)",
        line_color=color,
    )

    method_specs = [
        {
            "label": "2–50 μm (product N)",
            "color": "#1f77b4",
            "linestyle": "-",
            "cols": {
                "N": "n_drop_cm3",
                "LWC": "LWC_manualspec_g_m3",
                "Rv": "r_v_manualspec_um",
                "Reff": "r_eff_manualspec_um",
            },
        },
        {
            "label": "D≥2 μm, no upper (product N)",
            "color": "#d62728",
            "linestyle": "-",
            "cols": {
                "N": "n_drop_ge2_no_upper_cm3",
                "LWC": "LWC_ge2_no_upper_manualspec_g_m3",
                "Rv": "r_v_ge2_no_upper_manualspec_um",
                "Reff": "r_eff_ge2_no_upper_manualspec_um",
            },
        },
        {
            "label": "D≥2 μm, no upper (manual N)",
            "color": "#2ca02c",
            "linestyle": "--",
            "cols": {
                "N": "n_drop_ge2_no_upper_manual_cm3",
                "LWC": "LWC_ge2_no_upper_manualdirect_g_m3",
                "Rv": "r_v_ge2_no_upper_manualdirect_um",
                "Reff": "r_eff_ge2_no_upper_manualdirect_um",
            },
        },
    ]
    plot_three_method_compare_2x2(
        df_1min,
        f"{case_name}_compare_2x2_three_methods_1min.png",
        method_specs,
    )


def make_compare_outputs(results):
    if not results:
        return
    os.makedirs(COMPARE_DIRNAME, exist_ok=True)
    plot_compare_2x2(
        results,
        os.path.join(COMPARE_DIRNAME, "compare_2x2_1min_N_ylog.png"),
        n_logy=True,
        ycols=["n_drop_all_cm3", "LWC_g_m3", "Rv_um", "Reff_um"],
        ylabels=["N (cm$^{-3}$)", "LWC (g m$^{-3}$)", "R$_v$ (µm)", "R$_{eff}$ (µm)"]
    )
    plot_compare_2x2(
        results,
        os.path.join(COMPARE_DIRNAME, "compare_2x2_1min_N_ylinear.png"),
        n_logy=False,
        ycols=["n_drop_all_cm3", "LWC_g_m3", "Rv_um", "Reff_um"],
        ylabels=["N (cm$^{-3}$)", "LWC (g m$^{-3}$)", "R$_v$ (µm)", "R$_{eff}$ (µm)"]
    )
    plot_compare_2x2(
        results,
        os.path.join(COMPARE_DIRNAME, "compare_2x2_manualspec_1min.png"),
        n_logy=False,
        ycols=["n_drop_cm3", "LWC_manualspec_g_m3", "r_v_manualspec_um", "r_eff_manualspec_um"],
        ylabels=["N (cm$^{-3}$)", "LWC (g m$^{-3}$)", "R$_v$ (µm)", "R$_{eff}$ (µm)"]
    )
    plot_compare_2x2(
        results,
        os.path.join(COMPARE_DIRNAME, "compare_2x2_Dge2um_no_upper_1min.png"),
        n_logy=False,
        ycols=[
            "n_drop_ge2_no_upper_cm3",
            "LWC_ge2_no_upper_manualspec_g_m3",
            "r_v_ge2_no_upper_manualspec_um",
            "r_eff_ge2_no_upper_manualspec_um",
        ],
        ylabels=["N (cm$^{-3}$)", "LWC (g m$^{-3}$)", "R$_v$ (µm)", "R$_{eff}$ (µm)"]
    )
    plot_compare_2x2(
        results,
        os.path.join(COMPARE_DIRNAME, "compare_2x2_Dge2um_no_upper_manualN_1min.png"),
        n_logy=False,
        ycols=[
            "n_drop_ge2_no_upper_manual_cm3",
            "LWC_ge2_no_upper_manualdirect_g_m3",
            "r_v_ge2_no_upper_manualdirect_um",
            "r_eff_ge2_no_upper_manualdirect_um",
        ],
        ylabels=["N (cm$^{-3}$)", "LWC (g m$^{-3}$)", "R$_v$ (µm)", "R$_{eff}$ (µm)"]
    )
    plot_compare_mean_spectrum(
        results,
        os.path.join(COMPARE_DIRNAME, "compare_mean_spectrum_2to50um_1minavg.png"),
        spectrum_key="spectrum",
    )
    plot_compare_mean_spectrum(
        results,
        os.path.join(COMPARE_DIRNAME, "compare_mean_spectrum_Dge2um_no_upper_1minavg.png"),
        spectrum_key="spectrum_ge2_no_upper",
    )
    plot_heatmap_grid(results, os.path.join(COMPARE_DIRNAME, "compare_heatmap_grid.png"), spectrum_key="spectrum")
    plot_heatmap_grid(results, os.path.join(COMPARE_DIRNAME, "compare_heatmap_grid_Dge2um_no_upper.png"),
                      spectrum_key="spectrum_ge2_no_upper")


def build_case_result(
    case_name,
    label,
    color,
    df_1min,
    spectrum_time_s,
    spectrum_n_cm3,
    spectrum_edges_um,
    spectrum_ge2_no_upper_time_s,
    spectrum_ge2_no_upper_n_cm3,
    spectrum_ge2_no_upper_edges_um,
):
    return {
        "case_name": case_name,
        "label": label,
        "title": label,
        "color": color,
        "summary_1min": _ensure_optional_summary_columns(df_1min),
        "spectrum": {
            "spectrum_time_s": np.asarray(spectrum_time_s, dtype=float),
            "spectrum_n_cm3": np.asarray(spectrum_n_cm3, dtype=float),
            "D_edges_um": np.asarray(spectrum_edges_um, dtype=float),
        },
        "spectrum_ge2_no_upper": {
            "spectrum_time_s": np.asarray(spectrum_ge2_no_upper_time_s, dtype=float),
            "spectrum_n_cm3": np.asarray(spectrum_ge2_no_upper_n_cm3, dtype=float),
            "D_edges_um": np.asarray(spectrum_ge2_no_upper_edges_um, dtype=float),
        },
    }


def run_case(case_name, injection_rate, output_enabled, r_dry_m, N_cm3, formulae, color, label):
    settings = Settings(
        enable_collisions=ENABLE_COLLISIONS,
        output_seeding_case=output_enabled,
        timestep=DT,
        t_max=T_MAX,
        parcel_volume=PARCEL_VOLUME,
        initial_total_pressure=P0,
        initial_temperature=T0,
        initial_RH=RH0,
        w_min=W_MIN,
        w_max=W_MAX,
        updraft_period=W_PERIOD,
        initial_r_dry_m=r_dry_m,
        initial_N_cm3=N_cm3,
        super_droplet_injection_rate=injection_rate,
        n_sd_seeding=100,
        formulae=formulae,
        initial_aerosol_kappa=INITIAL_AEROSOL_KAPPA,
        seeding_kappa=SEEDING_KAPPA,
    )
    dump_run_config_txt(case_name, settings, out_txt=f"{case_name}_run_config.txt")
    sim = Simulation(settings)
    out = sim.run(
        BASE_D_EDGES_UM,
        output_interval_s=SPECTRUM_INTERVAL_S,
        rv_radius_range_um=(1.0, 25.0),
        no_upper_min_d_um=NO_UPPER_MIN_D_UM,
        no_upper_bin_step_um=NO_UPPER_BIN_STEP_UM,
    )
    if not output_enabled:
        return None

    print(
        f"[INFO] {case_name}: max wet diameter = {out.get('max_diameter_um', np.nan):.3f} μm; "
        f"D>=2 μm no-upper spectrum last edge = {np.asarray(out['spectrum_ge2_no_upper_D_edges_um'], dtype=float)[-1]:.3f} μm"
    )

    _, df_1min, df_sup, df_num = save_summary_csv(out, case_name)
    save_spectrum_csv(out["spectrum_time_s"], out["spectrum_n_cm3"], BASE_D_EDGES_UM, f"{case_name}_droplet_spectrum.csv")
    save_spectrum_csv(
        out["spectrum_ge2_no_upper_time_s"],
        out["spectrum_ge2_no_upper_n_cm3"],
        out["spectrum_ge2_no_upper_D_edges_um"],
        f"{case_name}_Dge2um_no_upper_droplet_spectrum.csv",
    )
    make_all_plots(
        case_name,
        df_1min,
        df_sup,
        df_num,
        out["spectrum_time_s"],
        out["spectrum_n_cm3"],
        BASE_D_EDGES_UM,
        out["spectrum_ge2_no_upper_time_s"],
        out["spectrum_ge2_no_upper_n_cm3"],
        out["spectrum_ge2_no_upper_D_edges_um"],
        color=color,
    )
    return build_case_result(
        case_name, label, color, df_1min,
        out["spectrum_time_s"], out["spectrum_n_cm3"], BASE_D_EDGES_UM,
        out["spectrum_ge2_no_upper_time_s"], out["spectrum_ge2_no_upper_n_cm3"], out["spectrum_ge2_no_upper_D_edges_um"],
    )


def reuse_case(case_name, output_enabled, color, label):
    if not output_enabled:
        return None
    (
        df_1min, df_sup, df_num,
        spectrum_time_s, spectrum_n_cm3, spectrum_edges_um,
        spectrum_ge2_no_upper_time_s, spectrum_ge2_no_upper_n_cm3, spectrum_ge2_no_upper_edges_um,
    ) = load_saved_case_outputs(case_name)

    max_diameter_um = float(pd.to_numeric(df_1min.get("max_diameter_um", np.nan), errors="coerce").dropna().iloc[0])         if "max_diameter_um" in df_1min.columns and pd.to_numeric(df_1min["max_diameter_um"], errors="coerce").notna().any() else np.nan
    print(
        f"[INFO] {case_name} (reuse): max wet diameter = {max_diameter_um:.3f} μm; "
        f"D>=2 μm no-upper spectrum last edge = {float(np.asarray(spectrum_ge2_no_upper_edges_um, dtype=float)[-1]):.3f} μm"
    )

    make_all_plots(
        case_name,
        df_1min,
        df_sup,
        df_num,
        spectrum_time_s,
        spectrum_n_cm3,
        spectrum_edges_um,
        spectrum_ge2_no_upper_time_s,
        spectrum_ge2_no_upper_n_cm3,
        spectrum_ge2_no_upper_edges_um,
        color=color,
    )
    return build_case_result(
        case_name, label, color, df_1min,
        spectrum_time_s, spectrum_n_cm3, spectrum_edges_um,
        spectrum_ge2_no_upper_time_s, spectrum_ge2_no_upper_n_cm3, spectrum_ge2_no_upper_edges_um,
    )


def main():
    global COMPARE_DIRNAME

    base_output_dir, run_output_dir, compare_output_dir = prepare_output_directories(
        OUTPUT_BASE_DIR, ENABLE_COLLISIONS
    )
    COMPARE_DIRNAME = compare_output_dir
    os.chdir(run_output_dir)

    print(f"[INFO] Output base directory: {base_output_dir}")
    print(f"[INFO] Collision mode: {'ON' if ENABLE_COLLISIONS else 'OFF'}")
    print(f"[INFO] Current collision output directory: {run_output_dir}")
    print(f"[INFO] Compare output directory: {COMPARE_DIRNAME}")
    print(f"[INFO] REUSE_EXISTING_RESULTS_ONLY = {REUSE_EXISTING_RESULTS_ONLY}")
    print(f"[INFO] BASE_D_EDGES_UM = {BASE_D_EDGES_UM}")
    print(f"[INFO] NO_UPPER_MIN_D_UM = {NO_UPPER_MIN_D_UM}")
    print(f"[INFO] NO_UPPER_BIN_STEP_UM = {NO_UPPER_BIN_STEP_UM}")

    r_dry_m, N_cm3, Dc_nm, De_nm = read_aerosol_3rows_dNdlog10D(AER_FILE)
    pd.DataFrame({
        "bin_id": np.arange(len(Dc_nm)),
        "D_left_nm": De_nm[:-1],
        "D_right_nm": De_nm[1:],
        "D_center_nm": Dc_nm,
        "N_bin_cm-3": N_cm3,
    }).to_csv("aerosol_bins_and_N_input_to_SDM.csv", index=False)
    formulae = Formulae(seed=100)

    results = []
    if RUN_CONTROL:
        if REUSE_EXISTING_RESULTS_ONLY:
            res = reuse_case("control", True, color="#000000", label="control")
        else:
            res = run_case("control", injection_rate_no_seeding, True, r_dry_m, N_cm3, formulae, color="#000000", label="control")
        if res is not None:
            results.append(res)

    if RUN_SEEDING:
        if REUSE_EXISTING_RESULTS_ONLY:
            res = reuse_case("seeding", OUTPUT_SEEDING_CASE, color="#d62728", label="seeding")
        else:
            res = run_case("seeding", injection_rate_seeding, OUTPUT_SEEDING_CASE, r_dry_m, N_cm3, formulae, color="#d62728", label="seeding")
        if res is not None:
            results.append(res)

    make_compare_outputs(results)
    print("Done.")
    assert np.all(np.diff(BASE_D_EDGES_UM) > 0), "BASE_D_EDGES_UM must be strictly increasing!"


if __name__ == "__main__":
    main()
