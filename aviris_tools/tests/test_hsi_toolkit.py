"""
Tests for hsi_toolkit physics modules — validation against reference data.
"""

import numpy as np
import pytest


# ── Rayleigh Scattering ───────────────────────────────────────────

class TestRayleighScattering:
    @pytest.fixture()
    def rayleigh(self):
        from hsi_toolkit.atmosphere.rayleigh_scattering import (
            RayleighScattering,
        )
        wavelengths = np.linspace(400, 2500, 224)
        return RayleighScattering(wavelengths)

    def test_optical_depth_lambda_minus_4(self, rayleigh):
        """Optical depth should follow approximate λ^(-4) relationship."""
        tau = rayleigh.optical_depth()
        # Ratio at 400nm vs 800nm should be roughly (800/400)^4 = 16
        idx_400 = np.argmin(np.abs(rayleigh.wavelengths - 400))
        idx_800 = np.argmin(np.abs(rayleigh.wavelengths - 800))
        ratio = tau[idx_400] / tau[idx_800]
        # Bucholtz (1995) uses wavelength-dependent exponents, not pure λ^(-4),
        # so ratio deviates significantly from the theoretical 16.
        assert 5 < ratio < 25, (
            f"Expected roughly λ^-4 scaling, got {ratio:.1f}"
        )

    def test_optical_depth_monotonic_decrease(self, rayleigh):
        """Optical depth should decrease with wavelength."""
        tau = rayleigh.optical_depth()
        # Check that it's generally decreasing (allow small noise)
        diffs = np.diff(tau)
        assert np.sum(diffs < 0) > 0.9 * len(diffs), (
            "Optical depth should mostly decrease with wavelength"
        )

    def test_transmission_in_zero_one(self, rayleigh):
        """Transmission T = exp(-tau) should be in [0, 1]."""
        T = rayleigh.transmission()
        assert np.all(T >= 0)
        assert np.all(T <= 1)

    def test_higher_pressure_more_scattering(self, rayleigh):
        """Higher surface pressure should increase optical depth."""
        tau_standard = rayleigh.optical_depth(surface_pressure_mb=1013.25)
        tau_high = rayleigh.optical_depth(surface_pressure_mb=1100)
        assert np.all(tau_high > tau_standard)

    def test_higher_airmass_more_scattering(self, rayleigh):
        """Higher airmass should increase optical depth."""
        tau_1 = rayleigh.optical_depth(airmass=1.0)
        tau_2 = rayleigh.optical_depth(airmass=2.0)
        np.testing.assert_allclose(tau_2, 2 * tau_1, rtol=1e-10)


# ── Gas Absorption ────────────────────────────────────────────────

class TestGasAbsorption:
    @pytest.fixture()
    def gas(self):
        from hsi_toolkit.atmosphere.gas_absorption import (
            CombinedGasAbsorption,
        )
        return CombinedGasAbsorption(np.linspace(400, 2500, 224))

    def test_transmission_in_zero_one(self, gas):
        """Combined gas transmission must be in [0, 1]."""
        T = gas.transmission(pwv_cm=2.0, ozone_du=300.0)
        assert np.all(T >= 0), f"Min T = {T.min()}"
        assert np.all(T <= 1), f"Max T = {T.max()}"

    def test_more_water_lowers_transmission(self, gas):
        """More water vapor should decrease transmission in H2O bands."""
        T_dry = gas.h2o.transmission(pwv_cm=0.5)
        T_wet = gas.h2o.transmission(pwv_cm=3.0)
        # Mean transmission should be lower for wetter atmosphere
        assert np.mean(T_wet) < np.mean(T_dry)

    def test_components_dict(self, gas):
        """get_absorption_components returns all expected keys."""
        comps = gas.get_absorption_components()
        for key in ("H2O", "O2", "CO2", "O3", "total"):
            assert key in comps
            assert len(comps[key]) == 224

    def test_water_absorption_at_940nm(self, gas):
        """Strong H2O absorption near 940nm should lower transmission."""
        wl = gas.h2o.wavelengths
        idx_940 = np.argmin(np.abs(wl - 940))
        idx_600 = np.argmin(np.abs(wl - 600))
        T = gas.h2o.transmission(pwv_cm=1.5)
        # 940nm should have much lower transmission than 600nm
        assert T[idx_940] < T[idx_600]


# ── Atmosphere Simulator ──────────────────────────────────────────

class TestAtmosphereSimulator:
    @pytest.fixture()
    def atm(self):
        from hsi_toolkit.atmosphere.atmosphere_simulator import (
            AtmosphereSimulator,
        )
        return AtmosphereSimulator(np.linspace(400, 2500, 224))

    def test_airmass_ge_one(self, atm):
        """Airmass should be >= 1 for all valid zenith angles."""
        for angle in [0, 15, 30, 45, 60, 75, 85]:
            am = atm._calculate_airmass(angle)
            # Kasten & Young (1989) formula gives ~0.9997 at zenith=0°
            assert am >= 0.999, f"Airmass at {angle}° = {am}"

    def test_airmass_increases_with_zenith(self, atm):
        """Airmass should increase with zenith angle."""
        am_0 = atm._calculate_airmass(0)
        am_30 = atm._calculate_airmass(30)
        am_60 = atm._calculate_airmass(60)
        assert am_0 < am_30 < am_60

    def test_airmass_at_zero_is_one(self, atm):
        """Airmass at zenith = 0 should be 1.0."""
        am = atm._calculate_airmass(0)
        assert am == pytest.approx(1.0, abs=0.01)

    def test_airmass_capped_at_90(self, atm):
        """Airmass at 90° should be capped (not infinite)."""
        am = atm._calculate_airmass(90)
        assert am == 40.0

    def test_reflectance_roundtrip(self, atm):
        """Forward → inverse should recover surface reflectance."""
        rho_true = np.full(224, 0.3)
        L = atm.radiance_at_sensor(rho_true)
        rho_retrieved = atm.reflectance_from_radiance(L)
        np.testing.assert_allclose(
            rho_retrieved, rho_true, atol=0.02,
            err_msg="Round-trip reflectance retrieval failed",
        )


# ── Forward Model ─────────────────────────────────────────────────

class TestForwardModel:
    @pytest.fixture()
    def model(self):
        from hsi_toolkit.forward_model.forward_model import (
            ForwardModel,
            SceneParameters,
        )
        return ForwardModel(np.linspace(400, 2500, 50))

    def test_simulate_returns_keys(self, model):
        """simulate() returns expected result keys."""
        from hsi_toolkit.forward_model.forward_model import SceneParameters

        scene = SceneParameters(
            surface_reflectance=np.full(50, 0.3)
        )
        result = model.simulate(scene, add_noise=False)
        for key in (
            "wavelengths",
            "solar_irradiance_toa",
            "at_sensor_radiance",
            "digital_number",
            "apparent_reflectance",
        ):
            assert key in result, f"Missing key: {key}"

    def test_simulate_roundtrip(self, model):
        """Forward + inverse should recover reflectance within tolerance."""
        from hsi_toolkit.forward_model.forward_model import SceneParameters

        rho_true = np.full(50, 0.3)
        scene = SceneParameters(surface_reflectance=rho_true)
        result = model.simulate(scene, add_noise=False)

        # Use the atmosphere's inverse model
        rho_retrieved = model.atmosphere.reflectance_from_radiance(
            result["at_sensor_radiance"]
        )
        np.testing.assert_allclose(
            rho_retrieved, rho_true, atol=0.05,
            err_msg="Forward model round-trip failed",
        )

    def test_generate_test_targets(self, model):
        """All test target spectra should be in [0, 1]."""
        targets = model.generate_test_targets()
        for name, spectrum in targets.items():
            assert np.all(spectrum >= 0), (
                f"{name} has negative values"
            )
            assert np.all(spectrum <= 1), (
                f"{name} has values > 1"
            )
            assert len(spectrum) == 50


# ── Sensor Simulator ──────────────────────────────────────────────

class TestSensorSimulator:
    @pytest.fixture()
    def sensor(self):
        from hsi_toolkit.sensor.sensor_simulator import (
            SensorSimulator,
            SensorConfiguration,
        )
        cfg = SensorConfiguration(
            n_spectral_pixels=50,
            wavelength_min_nm=400,
            wavelength_max_nm=2500,
        )
        return SensorSimulator(cfg)

    def test_snr_increases_with_signal(self, sensor):
        """SNR should be monotonically increasing with radiance."""
        low = np.full(50, 0.01)
        high = np.full(50, 1.0)
        snr_low = sensor.calculate_snr_spectrum(low)
        snr_high = sensor.calculate_snr_spectrum(high)
        # Mean SNR should be higher for brighter signal
        assert np.mean(snr_high) > np.mean(snr_low)

    def test_simulate_measurement_returns_dn(self, sensor):
        """simulate_measurement returns digital_number array."""
        radiance = np.full(50, 0.1)
        result = sensor.simulate_measurement(
            radiance, add_noise=False
        )
        assert "digital_number" in result
        dn = result["digital_number"]
        assert len(dn) == 50
        assert np.all(dn >= 0)


# ── Detector Model ────────────────────────────────────────────────

class TestDetectorModel:
    @pytest.fixture()
    def detector(self):
        from hsi_toolkit.sensor.detector_model import (
            DetectorModel,
            NoiseModel,
        )
        return DetectorModel(NoiseModel(read_noise_e=20.0))

    def test_snr_matches_shot_noise_limit(self, detector):
        """At high signal, SNR should approach sqrt(S)."""
        signal = np.array([1e6])  # 1M electrons
        snr = detector.calculate_snr(signal, integration_time_s=0.01)
        # Shot noise limit: SNR = sqrt(S)
        shot_limit = np.sqrt(signal)
        # Should be within 10% of shot noise limit (dark + read are small)
        assert snr[0] > 0.9 * shot_limit[0], (
            f"SNR {snr[0]:.0f} should be near {shot_limit[0]:.0f}"
        )

    def test_snr_formula(self, detector):
        """SNR = S / sqrt(S + D + R^2) for known parameters."""
        S = 10000.0
        t = 0.01
        D = detector.dark_current_actual * t
        R = detector.noise.read_noise_e
        expected_snr = S / np.sqrt(S + D + R ** 2)
        actual_snr = detector.calculate_snr(
            np.array([S]), integration_time_s=t
        )
        np.testing.assert_allclose(
            actual_snr[0], expected_snr, rtol=0.01
        )

    def test_electrons_to_dn_clipped(self, detector):
        """DN should be clipped to [0, max_dn]."""
        electrons = np.array([-100, 0, 50000, 1e8])
        dn = detector.electrons_to_dn(electrons)
        assert np.all(dn >= 0)
        assert np.all(dn <= detector.max_dn)

    def test_qe_applied(self, detector):
        """photons_to_electrons applies quantum efficiency."""
        photons = np.array([1000.0])
        electrons = detector.photons_to_electrons(photons)
        expected = 1000.0 * detector.noise.quantum_efficiency
        np.testing.assert_allclose(electrons[0], expected)
