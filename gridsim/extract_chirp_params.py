#!/usr/bin/env python3

"""
This file extracts the evolution of the instantaneous frequency and amplitude
of chirps from a real labeled dataset. The extracted parameters can then 
be used to simulate chirps with the same characteristics.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from eod.communication import chirp_model
from IPython import embed
from rich import print
from rich.progress import track
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

from gridtools.utils.filters import bandpass_filter
from gridtools.utils.new_datasets import ChirpData, Dataset, RawData, WavetrackerData
from gridtools.utils.transforms import envelope, instantaneous_frequency


def gaussian(
    x: np.ndarray, mu: float, height: float, width: float, kurt: float
) -> np.ndarray:
    chirp_sig = 0.5 * width / (2.0 * np.log(10.0)) ** (0.5 / kurt)
    curve = height * np.exp(-0.5 * (((x - mu) / chirp_sig) ** 2.0) ** kurt)
    return curve


def chirp_model(x, m1, h1, w1, k1, m2, h2, w2, k2, m3, h3, w3, k3):
    g1 = gaussian(x, m1, h1, w1, k1)
    g2 = gaussian(x, m2, h2, w2, k2)
    g3 = gaussian(x, m3, h3, w3, k3)
    return g1 + g2 + g3


import numpy as np
from scipy.signal import find_peaks


def initial_params(x, y):
    """
    Compute initial parameter estimates for the chirp model.

    Parameters
    ----------
    x : array-like
        Time axis.
    y : array-like
        Signal.

    Returns
    -------
    tuple
        Initial parameter estimates and boundaries.
    """
    # Compute the mean and standard deviation of the signal
    y_mean = np.mean(y)
    y_std = np.std(y)

    # Find the indices of the peaks and troughs
    peaks, _ = find_peaks(y, height=y_mean + 2 * y_std)
    troughs, _ = find_peaks(-y, height=-y_mean - 2 * y_std)

    # Check if any peaks or troughs were found
    if len(peaks) == 0 or len(troughs) == 0:
        return [
            y_mean,
            y_mean / 2,
            1,
            2,
            x[0],
            y_mean,
            y_mean / 2,
            1,
            2,
            x[-1],
            y_mean,
            y_mean / 2,
        ]

    # Sort the peaks and troughs by their x values
    peak_x = x[peaks]
    peak_y = y[peaks]
    trough_x = x[troughs]
    trough_y = y[troughs]
    peak_order = np.argsort(peak_y)
    trough_order = np.argsort(trough_x)
    peak_x = peak_x[peak_order]
    peak_y = peak_y[peak_order]
    trough_x = trough_x[trough_order]
    trough_y = trough_y[trough_order]

    # Compute the initial parameter estimates
    m1 = 0  # Center of the first Gaussian
    h1 = peak_y[-1] - trough_y[-1]  # Amplitude of the first Gaussian
    w1 = (peak_x[-1] - trough_x[-1]) / 5  # Standard deviation of the first Gaussian
    k1 = 1  # Kurtosis of the first Gaussian
    m2 = 0 + abs(w1) * 0.6  # Center of the second Gaussian
    h2 = -h1 / 5  # Amplitude of the second Gaussian
    w2 = w1 / 2  # Standard deviation of the second Gaussian
    k2 = 1  # Kurtosis of the second Gaussian
    m3 = peak_x[-1] + (peak_x[-1] - trough_x[-1]) / 10  # Center of the third Gaussian
    h3 = np.abs(h2)  # Amplitude of the third Gaussian
    w3 = w1 / 2  # Standard deviation of the third Gaussian
    k3 = 1  # Kurtosis of the third Gaussian

    return [m1, h1, w1, k1, m2, h2, w2, k2, m3, h3, w3, k3]


def get_upper_fish(dataset):
    min_fs = []
    track_ids = np.unique(dataset.track.idents[~np.isnan(dataset.track.idents)])
    for track_id in track_ids:
        f = dataset.track.freqs[dataset.track.idents == track_id]
        min_fs.append(np.min(f))
    return track_ids[np.argmax(min_fs)]


def get_next_lower_fish(dataset, upper_fish):
    min_fs = []
    track_ids = np.unique(dataset.track.idents[~np.isnan(dataset.track.idents)])
    for track_id in track_ids:
        if track_id == upper_fish:
            min_fs.append(np.inf)
        else:
            f = dataset.track.freqs[dataset.track.idents == track_id]
            min_fs.append(np.min(f))
    return track_ids[np.argmin(min_fs)]


def get_lower_fish_freq(dataset, chirp, lower_fish):
    track_freqs = dataset.track.freqs[dataset.track.idents == lower_fish]
    track_times = dataset.track.times[
        dataset.track.indices[dataset.track.idents == lower_fish]
    ]
    track_index = np.argmin(np.abs(track_times - chirp))
    return track_freqs[track_index]


def extract_features(data):
    time_window = 0.5

    upper_fish = get_upper_fish(data)
    lower_fish = get_next_lower_fish(data, upper_fish)
    chirp_times = data.chirp.times[data.chirp.idents == upper_fish]
    track_freqs = data.track.freqs[data.track.idents == upper_fish]
    track_times = data.track.times[data.track.indices[data.track.idents == upper_fish]]
    track_powers = data.track.powers[
        data.track.indices[data.track.idents == upper_fish], :
    ]

    freqs = []
    envs = []
    for chirp in chirp_times:
        track_index = np.argmin(np.abs(track_times - chirp))
        track_freq = track_freqs[track_index]
        lower_fish_freq = get_lower_fish_freq(data, chirp, lower_fish)

        lower_bound = track_freq - lower_fish_freq - 50
        track_power = track_powers[track_index, :]
        best_electrode = np.argmax(track_power)

        start_index = int(np.round((chirp - time_window / 2) * data.rec.samplerate))
        stop_index = int(np.round((chirp + time_window / 2) * data.rec.samplerate))
        raw_index = np.arange(start_index, stop_index)

        raw = data.rec.raw[raw_index, best_electrode]
        tuk = tukey(len(raw), alpha=0.1)
        raw = raw * tuk

        raw = bandpass_filter(
            signal=raw,
            samplerate=data.rec.samplerate,
            lowf=track_freq - 40,
            highf=track_freq + 280,
        )

        raw = raw / np.max(np.abs(raw))
        rawtime = (np.arange(len(raw)) - len(raw) / 2) / data.rec.samplerate

        freq = instantaneous_frequency(
            signal=raw,
            samplerate=data.rec.samplerate,
            smoothing_window=5,
        )

        dist = np.histogram(freq, bins=100)
        mode = dist[1][np.argmax(dist[0])]

        freq = freq - mode
        freq = freq[1000:-1000]
        tuk = tukey(len(freq), alpha=0.3)
        freq = freq * tuk
        time = (np.arange(len(freq)) - len(freq) / 2) / data.rec.samplerate

        peak_height = np.percentile(freq, 95)
        peaks, _ = find_peaks(freq, height=peak_height)

        peak_index = np.argmin(np.abs(time[peaks] - chirp))
        peak = peaks[peak_index]

        # skip if peak it too close to edge
        if abs(time[peak]) > time_window / 4:
            continue

        if np.min(freq) < -80:
            continue

        # compute envelope as well
        renv = envelope(raw, samplerate=data.rec.samplerate, cutoff_frequency=100)

        # remove low frequency modulation
        env = bandpass_filter(
            signal=renv,
            samplerate=data.rec.samplerate,
            lowf=0.1,
            highf=100,
        )

        # cut off the edges of the envelope to remove tukey window
        env = env[1000:-1000]

        mode = np.histogram(env, bins=100)[1][np.argmax(np.histogram(env, bins=100)[0])]
        env = (env - mode) * tuk
        envtime = (np.arange(len(env)) - len(env) / 2) / data.rec.samplerate

        # detect anolaies (peaks and troughs) in the envelope
        absenv = np.abs(env)
        env_peaks, _ = find_peaks(absenv, height=np.percentile(absenv, 99))

        # get the peak closest to the chirp
        env_peak_index = np.argmin(np.abs(envtime[env_peaks] - chirp))

        # skip if peak it too close to edge
        if abs(envtime[env_peaks[env_peak_index]]) > time_window / 4:
            continue

        # check if the peak is close to the frequency peak
        if abs(envtime[env_peaks[env_peak_index]] - time[peak]) > 0.05:
            continue

        # center the chirp on the center index using the peak on the frequency

        # descend the peak in both directions until the frequency drops below 10
        left, right = peak, peak
        while freq[left] > 10:
            left += 1
        while freq[right] > 10:
            right -= 1

        # find the center between the flanks of the peak
        center = (right - left) // 2 + left

        roll = len(freq) // 2 - center
        freq = np.roll(freq, roll)

        # center the env on the freq peak as well
        env = np.roll(env, roll)

        tuk = tukey(len(freq), alpha=1)
        freq = freq * tuk
        env = (env * tuk) + 1

        height = np.max(freq)

        # check if there are multiple large peaks
        cp, _ = find_peaks(freq, prominence=height * 0.5)
        if len(cp) > 1:
            continue

        # fig, axs = plt.subplots(3, 1, sharex=True)
        # axs[0].plot(rawtime, raw)
        # axs[0].plot(rawtime, renv, color="red")
        # axs[1].plot(time, freq)
        # axs[1].plot(time[peak], freq[peak], "o")
        # axs[1].axhline(0, color="gray")
        # axs[2].plot(envtime, env)
        # axs[2].plot(envtime[env_peaks], env[env_peaks], "o")
        # axs[2].axhline(1, color="gray")
        # plt.show()

        # fig, ax = plt.subplots(2, 1, sharex=True)
        # ax[0].plot(freq)
        # ax[0].plot(cp, freq[cp], "o")
        # ax[1].plot(env)
        # ax[0].axvline(len(freq) // 2, color="gray", linestyle="--", lw=1)
        # ax[1].axvline(len(freq) // 2, color="gray", linestyle="--", lw=1)
        # ax[0].axhline(0, color="gray", linestyle="--", lw=1)
        # ax[1].axhline(1, color="gray", linestyle="--", lw=1)
        # plt.show()

        freqs.append(freq)
        envs.append(env)

    return freqs, envs


def fit_model(freqs, envs):
    freq_fit = []
    for freq, env in track(
        zip(freqs, envs), description="Fitting chirps", total=len(freqs)
    ):
        # fig, ax = plt.subplots(2, 1, sharex=True)

        x = (np.arange(len(freq)) / 20000) - (len(freq) / 20000 / 2)

        p0_c = initial_params(x, freq)
        try:
            popt_c, pcov_c = curve_fit(
                f=chirp_model,
                xdata=x,
                ydata=freq,
                maxfev=100000,
                p0=p0_c,
            )
            freq_fit.append(popt_c)
            print("success")
        except:
            # ax[0].plot(x, freq, color="black")
            # ax[0].plot(x, chirp_model(x, *p0_c), color="blue", alpha=0.5)
            # ax[1].plot(x, env, color="black")
            # plt.show()
            freq_fit.append(np.full_like(p0_c, np.nan))
            print("fail")
            continue

        # ax[0].plot(x, freq, color="black", label="data")
        # ax[0].plot(x, chirp_model(x, *popt_c), color="red", label="fit")
        # ax[0].plot(x, chirp_model(x, *p0_c), color="blue", label="initial", alpha=0.5)
        # ax[1].plot(x, env, color="black", label="data")
        # ax[0].legend()
        # plt.show()

    return freq_fit


def save_arrays(freqs, envs, freq_fit, args):
    filename = args.input.name
    path = args.output
    np.save(path / f"{filename}_freqs.npy", freqs)
    np.save(path / f"{filename}_envs.npy", envs)
    np.save(path / f"{filename}_freq_fit.npy", freq_fit)


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=pathlib.Path, help="Path to dataset.")
    parser.add_argument("--output", "-o", type=pathlib.Path, help="Path to output.")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    return args


def main():
    args = interface()
    raw = RawData(args.input)
    chirps = ChirpData(args.input, detector="gt")
    wavetracker = WavetrackerData(args.input)
    dataset = Dataset(
        path=args.input,
        track=wavetracker,
        rec=raw,
        chirp=chirps,
    )

    freqs, envs = extract_features(dataset)
    freq_fit = fit_model(freqs, envs)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(20, 20))
    for freq, env, fit in zip(freqs, envs, freq_fit):
        x = (np.arange(len(freq)) / 20000) - (len(freq) / 20000 / 2)
        ax[0].plot(x, freq, color="black", alpha=0.1)
        ax[1].plot(x, chirp_model(x, *fit), color="black", alpha=0.1)
        ax[2].plot(x, env, color="black", alpha=0.1)

    ax[0].axvline(0, color="gray", linestyle="--", lw=1)
    ax[0].axhline(0, color="gray", linestyle="--", lw=1)
    ax[1].axvline(0, color="gray", linestyle="--", lw=1)
    ax[1].axhline(0, color="gray", linestyle="--", lw=1)
    ax[2].axvline(0, color="gray", linestyle="--", lw=1)
    ax[2].axhline(1, color="gray", linestyle="--", lw=1)

    ax[0].set_ylabel("Frequency (Hz)")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Amplitude")

    ax[0].set_title("Instantaneous Frequency")
    ax[1].set_title("Fitted instantaneous frequency")
    ax[2].set_title("Envelope")

    plt.savefig(args.output / f"{args.input.name}_chirps.png")
    plt.show()

    save_arrays(freqs, envs, freq_fit, args)


if __name__ == "__main__":
    main()
