import numpy as np
from pystrict import strict 
from PySDM import Formulae
from PySDM.physics import si
from PySDM.initialisation.spectra import Lognormal
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity


def rho_dry_air(p: float, T: float) -> float:
    """rho = p / (Rd T)"""
    Rd = 287.05 * si.J / si.kg / si.K
    return p / (Rd * T)


def es_tetens(T: float) -> float:
    """Saturation vapor pressure over liquid water (Pa), Tetens-type. T in K."""
    Tc = (T / si.K) - 273.15
    es_hPa = 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))
    return es_hPa * si.hPa


# water vapour mixing ratio
def qv_from_RH(p: float, T: float, RH: float) -> float:
    """qv = eps * e / (p - e), e = RH * es"""
    eps = 0.622
    e = RH * es_tetens(T)
    return eps * e / (p - e)


@strict
class Settings:
    def __init__(
        self,
        *,
        # switches
        enable_collisions: bool = True,
        output_seeding_case: bool = True,

        # time control
        timestep: float = 0.25 * si.s,
        t_max: float = 3 * si.h,

        # parcel config (fog-like defaults; you can tune)
        parcel_volume: float = 1.0 * si.m**3,
        initial_total_pressure: float = 1018 * si.hPa,
        initial_temperature: float = 278.0 * si.K,
        initial_RH: float = 0.98,

        # fog-scale updraft profile
        w_min: float = 1e-4 * si.m / si.s,
        w_max: float = 0.005 * si.m / si.s,
        updraft_period: float = None,

        # aerosol input from doc (provided by run script)
        initial_r_dry_m: np.ndarray = None,   # [m]
        initial_N_cm3: np.ndarray = None,     # [cm^-3]

        # seeding control
        super_droplet_injection_rate: callable = None,
        n_sd_seeding: int = 100,

        # misc
        formulae: Formulae = None,
        initial_aerosol_kappa: float = 0.3,
        seeding_kappa: float = 0.8,
        rain_water_radius_threshold: float = 0.1 * si.mm,
    ):
        if formulae is None:
            formulae = Formulae(seed=0)
        self.formulae = formulae

        if initial_r_dry_m is None or initial_N_cm3 is None:
            raise ValueError("initial_r_dry_m and initial_N_cm3 must be provided by the run script.")

        self.enable_collisions = enable_collisions
        self.output_seeding_case = output_seeding_case

        self.timestep = timestep
        self.t_max = t_max

        # parcel thermodynamics
        self.initial_total_pressure = initial_total_pressure
        self.initial_temperature = initial_temperature
        self.initial_RH = initial_RH
        self.initial_water_vapour_mixing_ratio = qv_from_RH(
            self.initial_total_pressure, self.initial_temperature, self.initial_RH
        )

        # enforce dry-air volume = 1 m^3 (or user-defined)
        self.parcel_volume = parcel_volume
        self.rho0 = rho_dry_air(self.initial_total_pressure, self.initial_temperature)
        self.mass_of_dry_air = self.rho0 * self.parcel_volume

        # updraft profile (keeps original example style)
        self.w_min = w_min
        self.w_max = w_max
        self.updraft_period = self.t_max if updraft_period is None else updraft_period

        def updraft_func(t):
            if t <= self.updraft_period / 2:
                return self.w_min + (self.w_max - self.w_min) * np.sin(
                    2 * np.pi * t / self.updraft_period
                ) ** 2
            else:
                return self.w_min

        self.updraft = updraft_func

        # aerosol from doc: N[cm^-3] -> multiplicity[#/kg_dry]
        r = np.asarray(initial_r_dry_m, dtype=float)
        N_cm3 = np.asarray(initial_N_cm3, dtype=float)

        mask = N_cm3 > 0
        r = r[mask]
        N_cm3 = N_cm3[mask]

        N_per_m3 = N_cm3 * 1e6
        n_total = N_per_m3 * self.parcel_volume                 # # in parcel
        n_in_dv_per_kg = n_total / self.mass_of_dry_air          # #/kg dry air

        self.initial_r_dry = r
        self.initial_n_in_dv_per_kg = n_in_dv_per_kg
        self.n_sd_initial = int(r.size)

        # seeding (keep example-style lognormal)
        if super_droplet_injection_rate is None:
            super_droplet_injection_rate = lambda t: 0
        self.super_droplet_injection_rate = super_droplet_injection_rate
        self.n_sd_seeding = n_sd_seeding

        r_seed, n_seed_in_dv = ConstantMultiplicity(
            Lognormal(
                norm_factor=10 / si.mg * self.mass_of_dry_air,
                m_mode=1 * si.um,
                s_geom=1.1,
            )
        ).sample(n_sd=self.n_sd_seeding)

        v_dry = self.formulae.trivia.volume(radius=r_seed)
        self.seeded_particle_multiplicity = n_seed_in_dv
        self.seeded_particle_extensive_attributes = {
            "signed water mass": [0.0001 * si.ng] * self.n_sd_seeding,
            "dry volume": v_dry,
            "kappa times dry volume": seeding_kappa * v_dry,
        }

        self.initial_aerosol_kappa = initial_aerosol_kappa
        self.seeding_kappa = seeding_kappa
        self.rain_water_radius_threshold = rain_water_radius_threshold

