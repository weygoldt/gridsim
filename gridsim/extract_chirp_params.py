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
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

from gridtools.utils.filters import bandpass_filter
from gridtools.utils.new_datasets import (
    ChirpData,
    Dataset,
    RawData,
    WavetrackerData,
)
from gridtools.utils.transforms import envelope, instantaneous_frequency


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
            print(track_id, np.min(f))
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
    print(f"Upper fish: {upper_fish}")
    lower_fish = get_next_lower_fish(data, upper_fish)
    print(f"Lower fish: {get_next_lower_fish(data, upper_fish)}")
    chirp_times = data.chirp.times[data.chirp.idents == upper_fish]
    track_freqs = data.track.freqs[data.track.idents == upper_fish]
    track_times = data.track.times[
        data.track.indices[data.track.idents == upper_fish]
    ]
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

        start_index = int(
            np.round((chirp - time_window / 2) * data.rec.samplerate)
        )
        stop_index = int(
            np.round((chirp + time_window / 2) * data.rec.samplerate)
        )
        raw_index = np.arange(start_index, stop_index)

        raw = data.rec.raw[raw_index, best_electrode]
        tuk = tukey(len(raw), alpha=0.1)
        raw = raw * tuk

        raw = bandpass_filter(
            signal=raw,
            samplerate=data.rec.samplerate,
            lowf=track_freq - 50,
            highf=track_freq + 300,
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

        # compute envelope as well
        renv = envelope(
            raw, samplerate=data.rec.samplerate, cutoff_frequency=100
        )

        # remove low frequency modulation
        env = bandpass_filter(
            signal=renv,
            samplerate=data.rec.samplerate,
            lowf=5,
            highf=100,
        )

        # cut off the edges of the envelope to remove tukey window
        env = env[1000:-1000]

        mode = np.histogram(env, bins=100)[1][
            np.argmax(np.histogram(env, bins=100)[0])
        ]
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

        tuk = tukey(len(freq), alpha=0.3)
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
    heigths = []
    widths = []
    fit = []
    for freq, env in zip(freqs, envs):
        fig, ax = plt.subplots(2, 1, sharex=True)
        height = np.max(freq)

        # compute the width of the freq peak
        left, right = (
            np.argmin(np.abs(freq[: len(freq) // 2] - height * 0.2)),
            np.argmin(np.abs(freq[len(freq) // 2 :] - height * 0.2))
            + len(freq) // 2,
        )
        width = (right - left) / 20000
        undershoot = np.min(freq) / height
        widths.append(width)
        heigths.append(height)

        # try to fit the chirp model to the peak
        x = (np.arange(len(freq)) / 20000) - (len(freq) / 20000 / 2)
        p0 = [0, width / 2, width, width, height, undershoot, 1, 1]
        try:
            popt, pcov = curve_fit(
                f=chirp_model,
                xdata=x,
                ydata=freq,
                maxfev=100000,
                p0=p0,
            )
        except:
            ax[0].plot(x, freq, color="black")
            ax[1].plot(x, env, color="black")
            plt.show()

            popt = np.full_like(p0, np.nan)
            fit.append(popt)
            continue
        ax[0].plot(x, freq, color="black")
        ax[0].plot(x, chirp_model(x, *popt), color="red")
        ax[1].plot(x, env, color="black")
        plt.show()
        fit.append(popt)

    plt.scatter(heigths, widths)
    plt.xlabel("Height")
    plt.ylabel("Width")
    plt.show()

    return fit


def save_arrays(freqs, envs, freq_fit, args):
    filename = args.input.name
    path = args.output
    np.save(path / f"{filename}_freqs.npy", freqs)
    np.save(path / f"{filename}_envs.npy", envs)
    np.save(path / f"{filename}_freq_fit.npy", freq_fit)


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=pathlib.Path, help="Path to dataset."
    )
    parser.add_argument(
        "--output", "-o", type=pathlib.Path, help="Path to output."
    )
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
    save_arrays(freqs, envs, freq_fit, args)


if __name__ == "__main__":
    main()
