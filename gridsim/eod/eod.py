#!/usr/bin/env python3

"""
Simulate eod wave type weakly electric fish. 
Most of the code is just slightly modified from the original code by
Jan Benda et al. in the thunderfish package. The original code can be found
here: https://github.com/janscience/thunderfish
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def eod(
    amplitudes: np.ndarray,
    phases: np.ndarray,
    frequency: Union[float, np.ndarray],
    samplerate: float,
    duration: float,
    phase0: float,
    noise_std: float,
) -> np.ndarray:
    """Simulate EOD waveform of a wave-type fish.

    The waveform is constructed by superimposing sinewaves of integral
    multiples of the fundamental frequency - the fundamental and its
    harmonics. The fundamental frequency of the EOD (EODf) is given by
    `frequency`. With `fish` relative amplitudes and phases of the
    fundamental and its harmonics are specified.

    The generated waveform is `duration` seconds long and is sampled with
    `samplerate` Hertz. Gaussian white noise with a standard deviation of
    `noise_std` is added to the generated waveform.

    Parameters
    ----------
    fish : str or dict
        Specify relative amplitudes and phases of the fundamental and its harmonics.
        If string then take amplitudes and phases from the `wavefish_harmonics` dictionary.
        If dictionary then take amplitudes and phases from the 'amlitudes' and 'phases' keys.
    frequency : float or numpy.ndarray
        EOD frequency of the fish in Hertz. Either fixed number or array for
        time-dependent frequencies.
    samplerate : float
        Sampling rate in Hertz.
    duration : float
        Duration of the generated data in seconds. Only used if frequency is scalar.
    phase0 : float
        Phase offset of the EOD waveform in radians.
    noise_std : float
        Standard deviation of additive Gaussian white noise.

    Returns
    -------
    data : numpy.ndarray
        Generated data of a wave-type fish.

    Raises
    ------
    KeyError:
        Unknown fish.
    IndexError:
        Amplitudes and phases differ in length.
    """

    # compute phase:
    if np.isscalar(frequency):
        phase = np.arange(0, duration, 1.0 / samplerate)
        phase *= frequency
    else:
        phase = np.cumsum(frequency) / samplerate

    # generate EOD:
    data = np.zeros(len(phase))
    for har, (ampl, phi) in enumerate(zip(amplitudes, phases)):
        if np.isfinite(ampl) and np.isfinite(phi):
            data += ampl * np.sin(
                2 * np.pi * (har + 1) * phase + phi - (har + 1) * phase0
            )

    # add noise:
    data += noise_std * np.random.randn(len(data))

    return data


def main():
    amplitudes = [0.90062, 0.15311, 0.072049, 0.012609, 0.011708]
    phases = [1.3623, 2.3246, 0.9869, 2.6492, -2.6885]
    sig = eod(
        amplitudes=amplitudes,
        phases=phases,
        frequency=100.0,
        samplerate=44100.0,
        duration=0.05,
    )
    plt.plot(sig)
    plt.show()


if __name__ == "__main__":
    main()
