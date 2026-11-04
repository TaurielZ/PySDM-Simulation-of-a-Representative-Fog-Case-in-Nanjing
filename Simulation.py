import numpy as np

from PySDM import Builder, products
from PySDM.backends import CPU
from PySDM.environments import Parcel
from PySDM.dynamics import AmbientThermodynamics, Condensation, Coalescence, Seeding
from PySDM.dynamics.collisions.collision_kernels import Geometric
from PySDM.physics import si

from Settings import Settings



def compute_droplet_radius_from_water_mass(water_mass_kg: np.ndarray) -> np.ndarray:
    """r = (3m/(4πρw))^(1/3). Returns radius in meters."""
    rho_w = 1000.0 * si.kg / si.m**3
    return (3.0 * water_mass_kg / (4.0 * np.pi * rho_w)) ** (1.0 / 3.0)



def binned_spectrum_number_conc_cm3(
    *,
    D_um: np.ndarray,
    weights_per_m3: np.ndarray,
    D_edges_um: np.ndarray,
) -> np.ndarray:
    """Histogram: returns number concentration in each diameter bin [cm^-3]."""
    hist, _ = np.histogram(D_um, bins=D_edges_um, weights=weights_per_m3)
    return hist / 1e6  # m^-3 -> cm^-3


def equivalent_diameters_from_edges_um(D_edges_um: np.ndarray) -> np.ndarray:
    """Geometric-mean equivalent diameter for each bin defined by D_edges_um."""
    D_edges_um = np.asarray(D_edges_um, dtype=float)
    if D_edges_um.ndim != 1 or D_edges_um.size < 2:
        raise ValueError("D_edges_um must be a 1D array with at least 2 edges")
    if not np.all(np.isfinite(D_edges_um)):
        raise ValueError("D_edges_um contains non-finite values")
    if not np.all(D_edges_um > 0):
        raise ValueError("D_edges_um must be strictly positive for geometric-mean centers")
    if not np.all(np.diff(D_edges_um) > 0):
        raise ValueError("D_edges_um must be strictly increasing")
    return np.sqrt(D_edges_um[:-1] * D_edges_um[1:])



def extend_bin_edges_by_fixed_step(
    D_edges_um: np.ndarray,
    target_max_um: float,
    step_um: float = 3.0,
) -> np.ndarray:
    """
    Extend upper diameter-bin edges by a fixed step until target_max_um is covered.
    The input edges must already be strictly increasing.
    """
    edges = np.asarray(D_edges_um, dtype=float).copy()
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("D_edges_um must be a 1D array with at least 2 edges")
    if not np.all(np.isfinite(edges)):
        raise ValueError("D_edges_um contains non-finite values")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("D_edges_um must be strictly increasing")
    if not np.isfinite(target_max_um):
        return edges
    if step_um <= 0:
        raise ValueError("step_um must be positive")
    while target_max_um > edges[-1] + 1e-12:
        edges = np.append(edges, edges[-1] + step_um)
    return edges



def manual_bulk_from_binned_spectrum(spec_n_cm3: np.ndarray, D_equivalent_um: np.ndarray):
    """
    Compute LWC, Rv, Reff from binned spectrum using equivalent diameters.
    Inputs:
      spec_n_cm3      number concentration in each diameter bin [cm^-3]
      D_equivalent_um equivalent diameter for each bin center [um]
    Returns:
      lwc_g_m3_manual, rv_um_manual, reff_um_manual
    Formulas:
      LWC  = 1e-6 * rho_w * sum((pi/6) * D^3 * n(D))
      Rv   = (sum(n * r^3) / Nf)^(1/3), r = D/2
      Reff = sum(n * r^3) / sum(n * r^2), r = D/2
    with rho_w = 1 g cm^-3, D/r in um, n in cm^-3.
    """
    n_cm3 = np.asarray(spec_n_cm3, dtype=float)
    D_um = np.asarray(D_equivalent_um, dtype=float)
    mask = np.isfinite(n_cm3) & np.isfinite(D_um) & (n_cm3 >= 0)
    if not np.any(mask):
        return float('nan'), float('nan'), float('nan')

    n_cm3 = n_cm3[mask]
    D_um = D_um[mask]
    r_um = 0.5 * D_um
    Nf = np.sum(n_cm3)

    lwc_g_m3 = float(1.0e-6 * np.sum((np.pi / 6.0) * (D_um ** 3) * n_cm3))

    if Nf > 0:
        rv_um = float((np.sum(n_cm3 * (r_um ** 3)) / Nf) ** (1.0 / 3.0))
    else:
        rv_um = float('nan')

    denom = np.sum(n_cm3 * (r_um ** 2))
    if denom > 0:
        reff_um = float(np.sum(n_cm3 * (r_um ** 3)) / denom)
    else:
        reff_um = float('nan')

    return lwc_g_m3, rv_um, reff_um


def manual_bulk_direct_from_particles(*, water_mass_kg: np.ndarray, r_um: np.ndarray, weights_per_m3: np.ndarray):
    """
    Compute N, LWC, Rv and Reff directly from particles satisfying a given filter.
    Inputs:
      water_mass_kg   per-particle liquid water mass [kg]
      r_um            wet radius [um]
      weights_per_m3  number concentration weights [m^-3]
    Returns:
      n_cm3, lwc_g_m3, rv_um, reff_um
    """
    water_mass_kg = np.asarray(water_mass_kg, dtype=float)
    r_um = np.asarray(r_um, dtype=float)
    weights_per_m3 = np.asarray(weights_per_m3, dtype=float)
    mask = (
        np.isfinite(water_mass_kg) & (water_mass_kg > 0)
        & np.isfinite(r_um) & (r_um > 0)
        & np.isfinite(weights_per_m3) & (weights_per_m3 > 0)
    )
    if not np.any(mask):
        return float('nan'), float('nan'), float('nan'), float('nan')

    water_mass_kg = water_mass_kg[mask]
    r_um = r_um[mask]
    weights_per_m3 = weights_per_m3[mask]

    n_cm3 = float(np.sum(weights_per_m3) / 1e6)
    lwc_g_m3 = float(np.sum(water_mass_kg * weights_per_m3) * 1e3)

    m0 = np.sum(weights_per_m3)
    if m0 > 0:
        rv_um = float((np.sum(weights_per_m3 * (r_um ** 3)) / m0) ** (1.0 / 3.0))
    else:
        rv_um = float('nan')

    denom = np.sum(weights_per_m3 * (r_um ** 2))
    if denom > 0:
        reff_um = float(np.sum(weights_per_m3 * (r_um ** 3)) / denom)
    else:
        reff_um = float('nan')

    return n_cm3, lwc_g_m3, rv_um, reff_um


def volume_mean_radius_um(
    *,
    r_m: np.ndarray,
    weights_per_m3: np.ndarray,
    r_min_m: float,
    r_max_m: float,
) -> float:
    """
    Volume-mean radius:
      Rv = (M3 / M0)^(1/3)
      M0 = sum(n)
      M3 = sum(r^3 n)
    Inputs:
      r_m            radius in meters
      weights_per_m3 number concentration weights in m^-3
      r_min_m/r_max_m radius bounds in meters
    Returns:
      Rv in micrometers (µm)
    """
    sel = (
        np.isfinite(r_m)
        & np.isfinite(weights_per_m3)
        & (weights_per_m3 > 0)
        & (r_m >= r_min_m)
        & (r_m <= r_max_m)
    )
    if not np.any(sel):
        return float("nan")

    M0 = np.sum(weights_per_m3[sel])
    if M0 <= 0:
        return float("nan")

    M3 = np.sum((r_m[sel] ** 3) * weights_per_m3[sel])
    return float((M3 / M0) ** (1.0 / 3.0) * 1e6)  # m -> µm



def saturation_vapour_pressure_tetens_pa(T_K: float) -> float:
    """Tetens formula, T in K, return saturation vapour pressure in Pa."""
    Tc = T_K - 273.15
    return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5)) * 100.0



def supersaturation_percent_from_qv_p_T(qv: float, p_pa: float, T_K: float) -> float:
    """
    Compute supersaturation (%) from water vapour mixing ratio qv [kg/kg],
    total pressure p [Pa], and temperature T [K].
    S(%) = (RH - 1) * 100, RH = e / es(T), e = qv p / (eps + qv)
    """
    eps = 0.622
    if not np.isfinite(qv) or not np.isfinite(p_pa) or not np.isfinite(T_K):
        return float("nan")
    if p_pa <= 0 or T_K <= 0:
        return float("nan")
    e_pa = qv * p_pa / (eps + qv)
    es_pa = saturation_vapour_pressure_tetens_pa(T_K)
    if es_pa <= 0:
        return float("nan")
    rh = e_pa / es_pa
    return float((rh - 1.0) * 100.0)



def _maybe_scalar(x):
    """Best-effort conversion of PySDM buffers/arrays/scalars to a float scalar."""
    if x is None:
        return None
    if hasattr(x, "get"):
        try:
            x = x.get()
        except Exception:
            pass
    if hasattr(x, "to_ndarray"):
        try:
            x = x.to_ndarray(raw=True)
        except Exception:
            try:
                x = x.to_ndarray()
            except Exception:
                pass
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    return float(arr.flat[0])



def _env_lookup_scalar(env, keys):
    """Try several likely access patterns for Parcel environment state."""
    for key in keys:
        # environment[key]
        try:
            val = env[key]
            val = _maybe_scalar(val)
            if val is not None and np.isfinite(val):
                return val
        except Exception:
            pass

        # environment.get(key)
        try:
            getter = getattr(env, "get", None)
            if getter is not None:
                val = getter(key)
                val = _maybe_scalar(val)
                if val is not None and np.isfinite(val):
                    return val
        except Exception:
            pass

        # environment.state[key]
        try:
            state = getattr(env, "state", None)
            if state is not None:
                val = state[key]
                val = _maybe_scalar(val)
                if val is not None and np.isfinite(val):
                    return val
        except Exception:
            pass

        # plain attribute, e.g. env.T or env.p
        try:
            if hasattr(env, key):
                val = getattr(env, key)
                val = _maybe_scalar(val)
                if val is not None and np.isfinite(val):
                    return val
        except Exception:
            pass

    return None



def try_get_supersaturation_percent(particulator) -> float:
    """
    Best-effort diagnosis of parcel supersaturation (%).
    We read current parcel-state qv/p/T from the environment if accessible,
    and then compute S=(RH-1)*100 ourselves.
    """
    env = getattr(particulator, "environment", None)
    if env is None:
        return float("nan")

    T_K = _env_lookup_scalar(env, [
        "T", "temperature", "air temperature", "ambient_temperature"
    ])
    p_pa = _env_lookup_scalar(env, [
        "p", "pressure", "air pressure", "ambient_pressure", "p0"
    ])
    qv = _env_lookup_scalar(env, [
        "water_vapour_mixing_ratio", "water vapor mixing ratio", "qv", "r_v"
    ])

    if T_K is None or p_pa is None or qv is None:
        return float("nan")

    return supersaturation_percent_from_qv_p_T(qv=qv, p_pa=p_pa, T_K=T_K)


def _dry_air_density_kg_m3_from_qv_p_T(qv: float, p_pa: float, T_K: float) -> float:
    """Dry-air density from parcel-state qv, p, T using dry-air partial pressure."""
    eps = 0.622
    Rd = 287.05 * si.J / (si.kg * si.K)

    if not np.isfinite(qv) or not np.isfinite(p_pa) or not np.isfinite(T_K):
        return float("nan")
    if p_pa <= 0 or T_K <= 0:
        return float("nan")

    e_pa = qv * p_pa / (eps + qv)
    pd_pa = p_pa - e_pa
    if pd_pa <= 0:
        return float("nan")

    return float(pd_pa / (Rd * T_K))


def try_get_parcel_volume_m3(particulator, settings) -> float:
    """
    Best-effort diagnosis of instantaneous parcel volume.

    Preferred path:
      V = mass_of_dry_air / rho_d
    where rho_d is diagnosed from the current parcel-state p, T, qv.

    Fallbacks:
      1) settings.parcel_volume
      2) settings.mass_of_dry_air / settings.rho0
    """
    env = getattr(particulator, "environment", None)

    T_K = _env_lookup_scalar(env, [
        "T", "temperature", "air temperature", "ambient_temperature"
    ]) if env is not None else None
    p_pa = _env_lookup_scalar(env, [
        "p", "pressure", "air pressure", "ambient_pressure", "p0"
    ]) if env is not None else None
    qv = _env_lookup_scalar(env, [
        "water_vapour_mixing_ratio", "water vapor mixing ratio", "qv", "r_v"
    ]) if env is not None else None

    mass_of_dry_air = _maybe_scalar(getattr(settings, "mass_of_dry_air", None))
    rho_d = _dry_air_density_kg_m3_from_qv_p_T(qv, p_pa, T_K)
    if mass_of_dry_air is not None and np.isfinite(mass_of_dry_air) and np.isfinite(rho_d) and rho_d > 0:
        return float(mass_of_dry_air / rho_d)

    parcel_volume = _maybe_scalar(getattr(settings, "parcel_volume", None))
    if parcel_volume is not None and np.isfinite(parcel_volume) and parcel_volume > 0:
        return float(parcel_volume)

    rho0 = _maybe_scalar(getattr(settings, "rho0", None))
    if mass_of_dry_air is not None and rho0 is not None and np.isfinite(mass_of_dry_air) and np.isfinite(rho0) and rho0 > 0:
        return float(mass_of_dry_air / rho0)

    return float("nan")


class Simulation:
    def __init__(self, settings: Settings):
        self.settings = settings

        builder = Builder(
            n_sd=settings.n_sd_initial + settings.n_sd_seeding,
            backend=CPU(formulae=settings.formulae, override_jit_flags={"parallel": False}),
            environment=Parcel(
                dt=settings.timestep,
                mass_of_dry_air=settings.mass_of_dry_air,
                w=settings.updraft,
                initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
                p0=settings.initial_total_pressure,
                T0=settings.initial_temperature,
            ),
        )

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())

        if settings.enable_collisions:
            builder.add_dynamic(Coalescence(collision_kernel=Geometric()))

        builder.add_dynamic(
            Seeding(
                super_droplet_injection_rate=settings.super_droplet_injection_rate,
                seeded_particle_multiplicity=settings.seeded_particle_multiplicity,
                seeded_particle_extensive_attributes=settings.seeded_particle_extensive_attributes,
            )
        )

        # init aerosol attributes from doc bins
        attributes = builder.particulator.environment.init_attributes(
            n_in_dv=settings.initial_n_in_dv_per_kg,
            kappa=settings.initial_aerosol_kappa,
            r_dry=settings.initial_r_dry,
        )

        # pad for seeded SDs (same style as example)为seeding的sd留空位
        self.particulator = builder.build(
            attributes={
                k: np.pad(
                    array=v,
                    pad_width=(0, settings.n_sd_seeding),
                    mode="constant",
                    constant_values=np.nan if k == "multiplicity" else 0,
                )
                for k, v in attributes.items()
            },
            products=(
                products.Time(),
                products.SuperDropletCountPerGridbox(name="sd_count"),
                products.EffectiveRadius(
                    name="r_eff",
                    unit="um",
                    radius_range=(1 * si.um, 25 * si.um),
                ),
                products.ParticleConcentration(
                    name="n_drop",
                    unit="cm^-3",
                    radius_range=(1 * si.um, 25 * si.um),
                ),
                products.ParticleConcentration(
                    name="n_drop_ge2_no_upper",
                    unit="cm^-3",
                    radius_range=(1 * si.um, np.inf),
                ),
                products.ParticleConcentration(
                    name="n_drop_all",
                    unit="cm^-3",
                ),
            ),
        )

        self.n_steps = int(settings.t_max // settings.timestep)


    def run(
        self,
        spectrum_D_edges_um,
        output_interval_s=60.0,
        rv_radius_range_um=(1.0, 25.0),
        no_upper_min_d_um=2.0,
        no_upper_bin_step_um=3.0,
    ):
        """
        Per-step outputs (dt):
          time_s, n_drop_cm3, n_drop_ge2_no_upper_cm3, LWC_g_m3, r_eff_um, r_v_um,
          supersaturation_pct, N_all_cm3, N_gt50_cm3,
          N_lt1um_cm3, N_1to25um_manual_cm3, N_gt25um_radius_cm3,
          LWC_manualspec_g_m3, r_v_manualspec_um, r_eff_manualspec_um,
          LWC_ge2_no_upper_manualspec_g_m3, r_v_ge2_no_upper_manualspec_um,
          r_eff_ge2_no_upper_manualspec_um,
          n_drop_ge2_no_upper_manual_cm3,
          LWC_ge2_no_upper_manualdirect_g_m3, r_v_ge2_no_upper_manualdirect_um,
          r_eff_ge2_no_upper_manualdirect_um

        Per-interval sampled outputs (NOT averaged, sampled at t=0,60,120...):
          summary_1min_*  (same times as spectrum)
          including the original 2–50 µm product-N/manual-spectrum bulk quantities,
          the D>=2 µm no-upper product-N/manual-spectrum bulk quantities, and
          the D>=2 µm no-upper direct-particle diagnostics

        Spectrum outputs:
          spectrum_time_s, spectrum_n_cm3                        (original fixed edges)
          spectrum_ge2_no_upper_time_s, spectrum_ge2_no_upper_n_cm3
          spectrum_ge2_no_upper_D_edges_um                      (dynamically extended)
          max_diameter_um                                       (global maximum wet diameter)
        """
        s = self.settings
        p = self.particulator

        out = dict(
            # dt outputs
            time_s=[],
            n_drop_cm3=[],
            n_drop_ge2_no_upper_cm3=[],
            n_drop_all_cm3=[],
            LWC_g_m3=[],
            r_eff_um=[],
            r_v_um=[],
            supersaturation_pct=[],
            N_all_cm3=[],
            N_gt50_cm3=[],
            N_lt1um_cm3=[],
            N_1to25um_manual_cm3=[],
            N_gt25um_radius_cm3=[],
            parcel_volume_m3=[],
            N_mult_all_count=[],
            N_wet_count=[],
            N_drop_all_count_from_product=[],
            LWC_manualspec_g_m3=[],
            r_v_manualspec_um=[],
            r_eff_manualspec_um=[],
            LWC_ge2_no_upper_manualspec_g_m3=[],
            r_v_ge2_no_upper_manualspec_um=[],
            r_eff_ge2_no_upper_manualspec_um=[],
            n_drop_ge2_no_upper_manual_cm3=[],
            LWC_ge2_no_upper_manualdirect_g_m3=[],
            r_v_ge2_no_upper_manualdirect_um=[],
            r_eff_ge2_no_upper_manualdirect_um=[],

            # sampled outputs (same time gate as spectrum)
            summary_1min_time_s=[],
            summary_1min_n_drop_cm3=[],
            summary_1min_n_drop_ge2_no_upper_cm3=[],
            summary_1min_n_drop_all_cm3=[],
            summary_1min_LWC_g_m3=[],
            summary_1min_r_eff_um=[],
            summary_1min_r_v_um=[],
            summary_1min_supersaturation_pct=[],
            summary_1min_N_all_cm3=[],
            summary_1min_N_gt50_cm3=[],
            summary_1min_N_lt1um_cm3=[],
            summary_1min_N_1to25um_manual_cm3=[],
            summary_1min_N_gt25um_radius_cm3=[],
            summary_1min_parcel_volume_m3=[],
            summary_1min_N_mult_all_count=[],
            summary_1min_N_wet_count=[],
            summary_1min_N_drop_all_count_from_product=[],
            summary_1min_LWC_manualspec_g_m3=[],
            summary_1min_r_v_manualspec_um=[],
            summary_1min_r_eff_manualspec_um=[],
            summary_1min_LWC_ge2_no_upper_manualspec_g_m3=[],
            summary_1min_r_v_ge2_no_upper_manualspec_um=[],
            summary_1min_r_eff_ge2_no_upper_manualspec_um=[],
            summary_1min_n_drop_ge2_no_upper_manual_cm3=[],
            summary_1min_LWC_ge2_no_upper_manualdirect_g_m3=[],
            summary_1min_r_v_ge2_no_upper_manualdirect_um=[],
            summary_1min_r_eff_ge2_no_upper_manualdirect_um=[],

            # spectrum outputs
            spectrum_time_s=[],
            spectrum_n_cm3=[],
            spectrum_ge2_no_upper_time_s=[],
            spectrum_ge2_no_upper_n_cm3=[],
        )

        next_output_time = 0.0
        D_edges_um = np.asarray(spectrum_D_edges_um, dtype=float)
        D_equivalent_um = equivalent_diameters_from_edges_um(D_edges_um)

        D_edges_ge2_no_upper_um = np.asarray(spectrum_D_edges_um, dtype=float).copy()
        if D_edges_ge2_no_upper_um[0] > no_upper_min_d_um + 1e-12:
            D_edges_ge2_no_upper_um = np.insert(D_edges_ge2_no_upper_um, 0, no_upper_min_d_um)
        elif abs(D_edges_ge2_no_upper_um[0] - no_upper_min_d_um) > 1e-12:
            D_edges_ge2_no_upper_um[0] = no_upper_min_d_um
        D_equivalent_ge2_no_upper_um = equivalent_diameters_from_edges_um(D_edges_ge2_no_upper_um)

        # convert rv range from µm -> m (radius bounds!)
        r_min_m = float(rv_radius_range_um[0]) * 1e-6
        r_max_m = float(rv_radius_range_um[1]) * 1e-6

        warned_supersat = False
        max_diameter_um = float("nan")

        for step in range(self.n_steps + 1):

            if step != 0:
                p.run(steps=1)

            # products from PySDM
            t = float(p.products["time"].get())
            n_drop = float(p.products["n_drop"].get())  # cm^-3
            n_drop_ge2_no_upper = float(p.products["n_drop_ge2_no_upper"].get())  # cm^-3
            n_drop_all = float(p.products["n_drop_all"].get())  # cm^-3
            r_eff = float(p.products["r_eff"].get())    # µm

            # SD attributes
            attrs = p.attributes
            water_mass = attrs["water mass"].to_ndarray(raw=True)       # kg
            multiplicity = attrs["multiplicity"].to_ndarray(raw=True)   # #/kg_dry

            mask_num = np.isfinite(multiplicity) & (multiplicity > 0)
            mask_wet = mask_num & np.isfinite(water_mass) & (water_mass > 0)

            # LWC: kg/kg_dry -> kg/m^3 -> g/m^3
            m_liq_per_kg = np.sum(water_mass[mask_wet] * multiplicity[mask_wet])
            lwc_g_m3 = float(m_liq_per_kg * s.rho0 * 1e3)

            # Parcel volume and parcel-total counts
            parcel_volume_m3 = try_get_parcel_volume_m3(p, s)

            mass_of_dry_air = _maybe_scalar(getattr(s, "mass_of_dry_air", None))
            count_factor = float(mass_of_dry_air) if mass_of_dry_air is not None and np.isfinite(mass_of_dry_air) and mass_of_dry_air > 0 else 1.0
            N_mult_all_count = float(np.sum(multiplicity[mask_num]) * count_factor)
            N_wet_count = float(np.sum(multiplicity[mask_wet]) * count_factor)

            # Total particle number concentration (all particles, regardless of current size)
            if np.isfinite(parcel_volume_m3) and parcel_volume_m3 > 0:
                weights_num_per_m3 = (multiplicity[mask_num] * count_factor) / parcel_volume_m3  # #/m^3
            else:
                weights_num_per_m3 = multiplicity[mask_num] * s.rho0  # fallback #/m^3
            N_all_cm3 = float(np.sum(weights_num_per_m3) / 1e6)
            N_drop_all_count_from_product = float(n_drop_all * 1e6 * parcel_volume_m3) if np.isfinite(parcel_volume_m3) else float("nan")

            # radii (m) from water mass for particles with positive liquid water
            water_mass_wet = water_mass[mask_wet]
            r_m = compute_droplet_radius_from_water_mass(water_mass_wet)
            r_um = r_m * 1e6
            if np.isfinite(parcel_volume_m3) and parcel_volume_m3 > 0:
                weights_wet_per_m3 = (multiplicity[mask_wet] * count_factor) / parcel_volume_m3  # #/m^3
            else:
                weights_wet_per_m3 = multiplicity[mask_wet] * s.rho0  # fallback #/m^3
            D_um = 2.0 * r_um

            if D_um.size:
                current_max_diameter_um = float(np.nanmax(D_um))
                max_diameter_um = current_max_diameter_um if not np.isfinite(max_diameter_um) else max(max_diameter_um, current_max_diameter_um)

            # N_gt50: number concentration of droplets with wet diameter > 50 um
            N_gt50_cm3 = float(np.sum(weights_wet_per_m3[D_um > 50.0]) / 1e6)

            # Manual diagnostics based on wet radius r_m:
            #   N_lt1um_cm3           : wet radius < 1 um
            #   N_1to25um_manual_cm3  : 1 <= wet radius <= 25 um
            #   N_gt25um_radius_cm3   : wet radius > 25 um
            sel_lt1 = r_um < 1.0
            sel_1to25 = (r_um >= 1.0) & (r_um <= 25.0)
            sel_gt25 = r_um > 25.0

            N_lt1um_cm3 = float(np.sum(weights_wet_per_m3[sel_lt1]) / 1e6)
            N_1to25um_manual_cm3 = float(np.sum(weights_wet_per_m3[sel_1to25]) / 1e6)
            N_gt25um_radius_cm3 = float(np.sum(weights_wet_per_m3[sel_gt25]) / 1e6)

            # Rv in µm, with radius bounds 1–25 µm
            r_v_um = volume_mean_radius_um(
                r_m=r_m,
                weights_per_m3=weights_wet_per_m3,
                r_min_m=r_min_m,
                r_max_m=r_max_m,
            )

            # Original fixed-edge spectrum / 2–50 µm manual bulk
            spec_n_cm3_dt = binned_spectrum_number_conc_cm3(
                D_um=D_um,
                weights_per_m3=weights_wet_per_m3,
                D_edges_um=D_edges_um,
            )
            lwc_manualspec_g_m3, r_v_manualspec_um, r_eff_manualspec_um = manual_bulk_from_binned_spectrum(
                spec_n_cm3=spec_n_cm3_dt,
                D_equivalent_um=D_equivalent_um,
            )

            # New D>=2 µm with no upper bound: dynamically extend edges by 3 µm above 50 µm
            sel_ge2 = np.isfinite(D_um) & (D_um >= no_upper_min_d_um)
            if np.any(sel_ge2):
                target_max_ge2_um = float(np.nanmax(D_um[sel_ge2]))
                new_edges = extend_bin_edges_by_fixed_step(
                    D_edges_ge2_no_upper_um,
                    target_max_um=target_max_ge2_um,
                    step_um=no_upper_bin_step_um,
                )
                if new_edges.size != D_edges_ge2_no_upper_um.size:
                    n_new_bins = new_edges.size - D_edges_ge2_no_upper_um.size
                    if out["spectrum_ge2_no_upper_n_cm3"]:
                        out["spectrum_ge2_no_upper_n_cm3"] = [
                            np.pad(np.asarray(arr, dtype=float), (0, n_new_bins), mode="constant", constant_values=0.0)
                            for arr in out["spectrum_ge2_no_upper_n_cm3"]
                        ]
                    D_edges_ge2_no_upper_um = new_edges
                    D_equivalent_ge2_no_upper_um = equivalent_diameters_from_edges_um(D_edges_ge2_no_upper_um)

            spec_ge2_no_upper_n_cm3_dt = binned_spectrum_number_conc_cm3(
                D_um=D_um,
                weights_per_m3=weights_wet_per_m3,
                D_edges_um=D_edges_ge2_no_upper_um,
            )
            (
                lwc_ge2_no_upper_manualspec_g_m3,
                r_v_ge2_no_upper_manualspec_um,
                r_eff_ge2_no_upper_manualspec_um,
            ) = manual_bulk_from_binned_spectrum(
                spec_n_cm3=spec_ge2_no_upper_n_cm3_dt,
                D_equivalent_um=D_equivalent_ge2_no_upper_um,
            )

            (
                n_drop_ge2_no_upper_manual_cm3,
                lwc_ge2_no_upper_manualdirect_g_m3,
                r_v_ge2_no_upper_manualdirect_um,
                r_eff_ge2_no_upper_manualdirect_um,
            ) = manual_bulk_direct_from_particles(
                water_mass_kg=water_mass_wet[sel_ge2],
                r_um=r_um[sel_ge2],
                weights_per_m3=weights_wet_per_m3[sel_ge2],
            )

            supersat_pct = try_get_supersaturation_percent(p)
            if (not warned_supersat) and (not np.isfinite(supersat_pct)):
                print("[WARN] Supersaturation could not be diagnosed from current Parcel environment state; output will be NaN.")
                warned_supersat = True

            # per-step outputs
            out["time_s"].append(t)
            out["n_drop_cm3"].append(n_drop)
            out["n_drop_ge2_no_upper_cm3"].append(n_drop_ge2_no_upper)
            out["n_drop_all_cm3"].append(n_drop_all)
            out["LWC_g_m3"].append(lwc_g_m3)
            out["r_eff_um"].append(r_eff)
            out["r_v_um"].append(r_v_um)
            out["supersaturation_pct"].append(supersat_pct)
            out["N_all_cm3"].append(N_all_cm3)
            out["N_gt50_cm3"].append(N_gt50_cm3)
            out["N_lt1um_cm3"].append(N_lt1um_cm3)
            out["N_1to25um_manual_cm3"].append(N_1to25um_manual_cm3)
            out["N_gt25um_radius_cm3"].append(N_gt25um_radius_cm3)
            out["parcel_volume_m3"].append(parcel_volume_m3)
            out["N_mult_all_count"].append(N_mult_all_count)
            out["N_wet_count"].append(N_wet_count)
            out["N_drop_all_count_from_product"].append(N_drop_all_count_from_product)
            out["LWC_manualspec_g_m3"].append(lwc_manualspec_g_m3)
            out["r_v_manualspec_um"].append(r_v_manualspec_um)
            out["r_eff_manualspec_um"].append(r_eff_manualspec_um)
            out["LWC_ge2_no_upper_manualspec_g_m3"].append(lwc_ge2_no_upper_manualspec_g_m3)
            out["r_v_ge2_no_upper_manualspec_um"].append(r_v_ge2_no_upper_manualspec_um)
            out["r_eff_ge2_no_upper_manualspec_um"].append(r_eff_ge2_no_upper_manualspec_um)
            out["n_drop_ge2_no_upper_manual_cm3"].append(n_drop_ge2_no_upper_manual_cm3)
            out["LWC_ge2_no_upper_manualdirect_g_m3"].append(lwc_ge2_no_upper_manualdirect_g_m3)
            out["r_v_ge2_no_upper_manualdirect_um"].append(r_v_ge2_no_upper_manualdirect_um)
            out["r_eff_ge2_no_upper_manualdirect_um"].append(r_eff_ge2_no_upper_manualdirect_um)

            # sampled outputs at 0,60,120... (instantaneous, not averaged)
            if t >= next_output_time - 1e-12:

                out["summary_1min_time_s"].append(t)
                out["summary_1min_n_drop_cm3"].append(n_drop)
                out["summary_1min_n_drop_ge2_no_upper_cm3"].append(n_drop_ge2_no_upper)
                out["summary_1min_n_drop_all_cm3"].append(n_drop_all)
                out["summary_1min_LWC_g_m3"].append(lwc_g_m3)
                out["summary_1min_r_eff_um"].append(r_eff)
                out["summary_1min_r_v_um"].append(r_v_um)
                out["summary_1min_supersaturation_pct"].append(supersat_pct)
                out["summary_1min_N_all_cm3"].append(N_all_cm3)
                out["summary_1min_N_gt50_cm3"].append(N_gt50_cm3)
                out["summary_1min_N_lt1um_cm3"].append(N_lt1um_cm3)
                out["summary_1min_N_1to25um_manual_cm3"].append(N_1to25um_manual_cm3)
                out["summary_1min_N_gt25um_radius_cm3"].append(N_gt25um_radius_cm3)
                out["summary_1min_parcel_volume_m3"].append(parcel_volume_m3)
                out["summary_1min_N_mult_all_count"].append(N_mult_all_count)
                out["summary_1min_N_wet_count"].append(N_wet_count)
                out["summary_1min_N_drop_all_count_from_product"].append(N_drop_all_count_from_product)
                out["summary_1min_LWC_manualspec_g_m3"].append(lwc_manualspec_g_m3)
                out["summary_1min_r_v_manualspec_um"].append(r_v_manualspec_um)
                out["summary_1min_r_eff_manualspec_um"].append(r_eff_manualspec_um)
                out["summary_1min_LWC_ge2_no_upper_manualspec_g_m3"].append(lwc_ge2_no_upper_manualspec_g_m3)
                out["summary_1min_r_v_ge2_no_upper_manualspec_um"].append(r_v_ge2_no_upper_manualspec_um)
                out["summary_1min_r_eff_ge2_no_upper_manualspec_um"].append(r_eff_ge2_no_upper_manualspec_um)
                out["summary_1min_n_drop_ge2_no_upper_manual_cm3"].append(n_drop_ge2_no_upper_manual_cm3)
                out["summary_1min_LWC_ge2_no_upper_manualdirect_g_m3"].append(lwc_ge2_no_upper_manualdirect_g_m3)
                out["summary_1min_r_v_ge2_no_upper_manualdirect_um"].append(r_v_ge2_no_upper_manualdirect_um)
                out["summary_1min_r_eff_ge2_no_upper_manualdirect_um"].append(r_eff_ge2_no_upper_manualdirect_um)

                out["spectrum_time_s"].append(t)
                out["spectrum_n_cm3"].append(np.asarray(spec_n_cm3_dt, dtype=float))
                out["spectrum_ge2_no_upper_time_s"].append(t)
                out["spectrum_ge2_no_upper_n_cm3"].append(np.asarray(spec_ge2_no_upper_n_cm3_dt, dtype=float))

                next_output_time += output_interval_s

        # finalize: convert to numpy arrays
        for k in (
            "time_s", "n_drop_cm3", "n_drop_ge2_no_upper_cm3", "n_drop_all_cm3", "LWC_g_m3",
            "r_eff_um", "r_v_um", "supersaturation_pct", "N_all_cm3", "N_gt50_cm3",
            "N_lt1um_cm3", "N_1to25um_manual_cm3", "N_gt25um_radius_cm3",
            "parcel_volume_m3", "N_mult_all_count", "N_wet_count", "N_drop_all_count_from_product",
            "LWC_manualspec_g_m3", "r_v_manualspec_um", "r_eff_manualspec_um",
            "LWC_ge2_no_upper_manualspec_g_m3", "r_v_ge2_no_upper_manualspec_um",
            "r_eff_ge2_no_upper_manualspec_um",
            "n_drop_ge2_no_upper_manual_cm3", "LWC_ge2_no_upper_manualdirect_g_m3",
            "r_v_ge2_no_upper_manualdirect_um", "r_eff_ge2_no_upper_manualdirect_um",
            "summary_1min_time_s", "summary_1min_n_drop_cm3", "summary_1min_n_drop_ge2_no_upper_cm3",
            "summary_1min_n_drop_all_cm3", "summary_1min_LWC_g_m3",
            "summary_1min_r_eff_um", "summary_1min_r_v_um",
            "summary_1min_supersaturation_pct", "summary_1min_N_all_cm3",
            "summary_1min_N_gt50_cm3", "summary_1min_N_lt1um_cm3",
            "summary_1min_N_1to25um_manual_cm3", "summary_1min_N_gt25um_radius_cm3",
            "summary_1min_parcel_volume_m3", "summary_1min_N_mult_all_count",
            "summary_1min_N_wet_count", "summary_1min_N_drop_all_count_from_product",
            "summary_1min_LWC_manualspec_g_m3", "summary_1min_r_v_manualspec_um",
            "summary_1min_r_eff_manualspec_um",
            "summary_1min_LWC_ge2_no_upper_manualspec_g_m3",
            "summary_1min_r_v_ge2_no_upper_manualspec_um",
            "summary_1min_r_eff_ge2_no_upper_manualspec_um",
            "summary_1min_n_drop_ge2_no_upper_manual_cm3",
            "summary_1min_LWC_ge2_no_upper_manualdirect_g_m3",
            "summary_1min_r_v_ge2_no_upper_manualdirect_um",
            "summary_1min_r_eff_ge2_no_upper_manualdirect_um",
            "spectrum_time_s", "spectrum_ge2_no_upper_time_s",
        ):
            out[k] = np.asarray(out[k], dtype=float)

        out["spectrum_n_cm3"] = np.asarray(out["spectrum_n_cm3"], dtype=float)
        out["spectrum_ge2_no_upper_n_cm3"] = np.asarray(out["spectrum_ge2_no_upper_n_cm3"], dtype=float)
        out["spectrum_ge2_no_upper_D_edges_um"] = np.asarray(D_edges_ge2_no_upper_um, dtype=float)
        out["max_diameter_um"] = float(max_diameter_um)

        return out
