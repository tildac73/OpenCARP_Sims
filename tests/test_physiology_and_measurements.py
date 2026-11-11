import numpy as np

from efov import config as cfg, measurements, physiology
from helpers import SMALL


def test_activation_plane_speed_and_direction():
    grid = np.array([[x, y, 0.0] for x in np.linspace(-4, 4, 9) for y in np.linspace(1, 6, 6)])
    times = 10 + grid[:, 0] / 0.9
    plane = physiology.activationPlane(grid, times, np.array([1.0, 0.0]))
    assert abs(plane["speedMPerS"] - 0.9) < 1e-9 and plane["directionErrorDeg"] < 1e-6
    toward = physiology.activationPlane(grid, 10 - grid[:, 1] / 0.9, np.array([0.0, -1.0]))
    assert toward["directionErrorDeg"] < 1e-6


def test_activation_times_interpolate_the_crossing():
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([[-80.0], [-40.0], [0.0]])
    first, count = physiology.activationTimes(t, v, -20.0)
    assert abs(first[0] - 1.5) < 1e-12 and count[0] == 1


def test_resampling_preserves_amplitude_and_timing():
    config = cfg.loadConfig(SMALL)
    t = np.arange(1000.0, 1400.0, 0.2)                      # 5 kHz
    signal = np.cos(2 * np.pi * 100 * (t - 1200) / 1000.0 + 0.3)        # phase offset: no crossing on a sample
    resampled, times = measurements.acquire(config, signal[None], t)
    assert np.isclose(np.diff(times)[0], 0.5)
    middle = slice(200, 600)
    expected = np.cos(2 * np.pi * 100 * (times[middle] - 1200) / 1000.0 + 0.3)
    # Timing: zero crossings line up exactly (zero-phase polyphase FIR).
    crossings = lambda x: np.flatnonzero(np.diff(np.sign(x)) != 0)
    np.testing.assert_array_equal(crossings(resampled[0, middle]), crossings(expected))
    # Amplitude: the default Kaiser filter has passband ripple (0.13 % measured at 100 Hz, 3 Nov 2026). Criterion:
    # well below the 5 % PP convergence tolerance (DL-2026-10-28-05), set at 0.5 %.
    assert np.max(np.abs(resampled[0, middle] - expected)) < 0.005


def syntheticEgm(config, windows):
    """Planar wave in +x past every electrode: a biphasic deflection at each electrode's arrival time."""
    from efov.electrograms import analysisTimes
    from efov.geometry import electrodeLayout
    layout = electrodeLayout(config)
    beatStart = (cfg.beatCount(config) - 1) * cfg.cycleLengthMs(config)
    times = np.round(np.arange(*analysisTimes(config, windows), 0.2), 6)
    times = np.append(times, round(times[-1] + 0.2, 6))
    arrival = beatStart + 20 + (layout["centres"][..., 0] + 8) / 0.9
    lesion = np.where(layout["centres"][..., 1] < 0, 0.05, 1.0)                   # smaller signals over the lesion
    dt = times[None, None, :] - arrival[..., None]
    uni = lesion[..., None] * -dt * np.exp(-(dt / 1.5) ** 2)
    return dict(uni=uni.astype(np.float32), timeMs=times, placements=layout["placements"], centres=layout["centres"])


def test_measurement_rows_in_model_units_with_per_placement_windows():
    config = cfg.loadConfig(SMALL)
    placements = len(cfg.centreYmm(config)) * len(config["values"]["gapsMm"])
    windows = np.tile([5.0, 45.0], (placements, 1))
    windows[1] += 3.0                                                               # one placement with a later window
    egm = syntheticEgm(config, windows)
    table, measured, clean, traceInfo, acquired = measurements.measureCondition(config, "lesion", egm, windows)
    bipoles = table[(table.kind == "bi") & (table.noiseRepeat == 0) & np.isclose(table.centreYmm, 0) & np.isclose(table.gapMm, 0.25)]
    assert len(bipoles) == 9 and np.median(bipoles.ppMv) > 0.5                     # physical units: unscaled synthetic amplitude
    differential = table[table.kind != "uni"]
    assert differential.pfHz.notna().any() and differential.pfGlobalHz.notna().any()
    assert set(differential.ppWindow) <= {"peakFrequency", "window"}
    assert table[table.kind == "uni"].pfHz.isna().all()
    assert set(table.kind) == {"uni", "bi", "omni"} and table.label.nunique() == 16 + 9 + 36
    beatStart = (cfg.beatCount(config) - 1) * cfg.cycleLengthMs(config)
    starts = table.groupby("placement").windowStartMs.first()
    assert np.isclose(starts[1] - starts[0], 3.0) and np.isclose(starts[0], beatStart + 5.0)
    assert measured.shape[1] == clean.shape[1] == 81                               # 40 ms at 2 kHz, both ends included
    assert measurements.filterHistoryCheck(config, acquired, windows)["doubleHistoryAvailable"]
    robustness = measurements.pfNoiseRobustness(table)
    assert {"pfHz_clean", "pfHz_noisy", "pfGlobalHz_clean", "pfGlobalHz_noisy"} <= set(robustness.columns)


def test_trailing_context_extends_the_segment_without_changing_filtered_signals():
    """FM-36: the segment runs trailingContextMs past the latest window; the causal band-pass output inside the windows is unchanged."""
    import copy
    import pytest
    from efov.electrograms import analysisTimes
    config = cfg.loadConfig(SMALL)
    placements = len(cfg.centreYmm(config)) * len(config["values"]["gapsMm"])
    windows = np.tile([5.0, 45.0], (placements, 1))
    egm = syntheticEgm(config, windows)
    context = config["fixed"]["analysis"]["trailingContextMs"]
    acquired, times = measurements.acquire(config, egm["uni"].astype(float), egm["timeMs"])
    short, starts, length = measurements.placementIndices(times, config, windows)
    long, longStarts, longLength = measurements.placementIndices(times, config, windows, context)
    assert long.start == short.start and np.array_equal(starts, longStarts) and length == longLength
    assert np.isclose(times[long][-1] - times[short][-1], context)
    without = copy.deepcopy(config); without["fixed"]["analysis"]["trailingContextMs"] = 0.0
    tableWith, _, cleanWith, _, _ = measurements.measureCondition(config, "lesion", egm, windows)
    tableWithout, _, cleanWithout, _, _ = measurements.measureCondition(without, "lesion", egm, windows)
    noiseFree = tableWith.traceIndex[tableWith.noiseRepeat == 0].to_numpy()        # noisy repeats draw noise for a longer segment
    np.testing.assert_array_equal(cleanWith[noiseFree], cleanWithout[noiseFree])   # same weights, same filtered window traces
    unipoles = (tableWith.kind == "uni") & (tableWith.noiseRepeat == 0)
    np.testing.assert_array_equal(tableWith.cleanPpMv[unipoles], tableWithout.cleanPpMv[unipoles])
    with pytest.raises(ValueError, match="trailing context"):
        measurements.placementIndices(times, config, windows, 10 * context)
    with pytest.raises(ValueError, match="past the end of the simulation"):
        analysisTimes(config, np.tile([5.0, 400.0], (placements, 1)))


def test_noise_only_bipolar_peak_to_peak():
    """FM-44: the noise floor of a bipole through the acquisition chain, in the units of the noise.

    Zero without noise; proportional to the noise SD (same seed); and between 4 and 6 SDs of the band-passed
    bipole for a 60 ms window (120 samples), whose SD is sqrt(2 x 0.472) x the electrode SD (noise-power
    gain of the 30-500 Hz band-pass at 2 kHz, decision log FM-36).
    """
    from efov import config as cfg
    from helpers import SMALL
    base = cfg.loadConfig(SMALL)
    assert measurements.noiseOnlyBipolarPpMv(cfg.loadConfig(SMALL, {"noiseSdMv": 0.0}), 120) == 0.0
    floor = measurements.noiseOnlyBipolarPpMv(base, 120)
    doubled = measurements.noiseOnlyBipolarPpMv(cfg.loadConfig(SMALL, {"noiseSdMv": 2 * base["values"]["noiseSdMv"]}), 120)
    assert np.isclose(doubled, 2 * floor, rtol=1e-12)
    bipoleSd = np.sqrt(2 * 0.472) * base["values"]["noiseSdMv"]
    assert 4 * bipoleSd < floor < 6 * bipoleSd
