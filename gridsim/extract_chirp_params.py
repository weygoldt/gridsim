#!/usr/bin

"""
This file extracts the evolution of the instantaneous frequency and amplitude
of chirps from a real labeled dataset. The extracted parameters can then 
be used to simulate chirps with the same characteristics.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from rich import print
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

from gridtools.utils.filters import bandpass_filter
from gridtools.utils.new_datasets import (
    ChirpData,
    Dataset,
    RawData,
    WavetrackerData,
)
from gridtools.utils.transforms import instantaneous_frequency


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
    time_window = 1

    upper_fish = get_upper_fish(data)
    lower_fish = get_next_lower_fish(data, upper_fish)
    chirp_times = data.chirp.times[data.chirp.idents == upper_fish]
    track_freqs = data.track.freqs[data.track.idents == upper_fish]
    track_times = data.track.times[
        data.track.indices[data.track.idents == upper_fish]
    ]
    track_powers = data.track.powers[
        data.track.indices[data.track.idents == upper_fish], :
    ]

    for chirp in chirp_times:
        track_index = np.argmin(np.abs(track_times - chirp))
        track_freq = track_freqs[track_index]
        lower_fish_freq = get_lower_fish_freq(data, chirp, lower_fish)

        if track_freq - lower_fish_freq < 100:
            print(
                f"Skipping chirp, too close to lower fish! df: {track_freq-lower_fish_freq}"
            )
            continue

        lower_bound = track_freq - lower_fish_freq - 20

        track_power = track_powers[track_index, :]
        best_electrode = np.argmax(track_power)

        start_index = int(
            np.round((chirp - time_window / 2) * data.recording.samplerate)
        )
        stop_index = int(
            np.round((chirp + time_window / 2) * data.recording.samplerate)
        )
        raw_index = np.arange(start_index, stop_index)

        raw = data.recording.raw[raw_index, best_electrode]
        raw = bandpass_filter(
            signal=raw,
            samplerate=data.recording.samplerate,
            lowf=track_freq - lower_bound,
            highf=track_freq * 1.5,
        )

        freq = instantaneous_frequency(
            signal=raw,
            samplerate=data.recording.samplerate,
            smoothing_window=5,
        )

        edges = int(np.round(0.3 * len(freq)))
        freq = freq[edges:-edges]

        dist = np.histogram(freq, bins=100)
        mode = dist[1][np.argmax(dist[0])]

        freq = freq - mode
        tukey_window = tukey(len(freq), alpha=0.3)
        freq = freq * tukey_window

        # center the chirp in the middle of the window
        peak_height = np.percentile(freq, 95)

        if peak_height < 20:
            continue

        peaks, _ = find_peaks(freq, height=peak_height)

        if len(peaks) > 1:
            continue

        plt.plot(freq)
        plt.plot(peaks, freq[peaks], "x")
        plt.show()


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=pathlib.Path, help="Path to dataset."
    )
    parser.add_argument(
        "--output", "-o", type=pathlib.Path, help="Path to output."
    )
    args = parser.parse_args()
    return args


def main():
    args = interface()
    raw = RawData(args.input)
    chirps = ChirpData(args.input)
    wavetracker = WavetrackerData(args.input)
    dataset = Dataset(
        path=args.input,
        track=wavetracker,
        recording=raw,
        chirp=chirps,
    )

    features = extract_features(dataset)


if __name__ == "__main__":
    main()
