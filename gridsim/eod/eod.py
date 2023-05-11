#!/usr/bin/env python3

"""
Simulate eod wave type weakly electric fish. 
Most of the code is just slightly modified from the original code by
Jan Benda et al. in the thunderfish package. The original code can be found
here: https://github.com/janscience/thunderfish
"""

from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

species_name = dict(
    Sine="Sinewave",
    Alepto="Apteronotus leptorhynchus",
    Arostratus="Apteronotus rostratus",
    Eigenmannia="Eigenmannia spec.",
    Sternarchella="Sternarchella terminalis",
    Sternopygus="Sternopygus dariensis",
)

Sine_harmonics = dict(amplitudes=(1.0,), phases=(0.5 * np.pi,))

Apteronotus_leptorhynchus_harmonics = dict(
    amplitudes=(0.90062, 0.15311, 0.072049, 0.012609, 0.011708),
    phases=(1.3623, 2.3246, 0.9869, 2.6492, -2.6885),
)

Apteronotus_rostratus_harmonics = dict(
    amplitudes=(
        0.64707,
        0.43874,
        0.063592,
        0.07379,
        0.040199,
        0.023073,
        0.0097678,
    ),
    phases=(2.2988, 0.78876, -1.316, 2.2416, 2.0413, 1.1022, -2.0513),
)

Eigenmannia_harmonics = dict(
    amplitudes=(1.0087, 0.23201, 0.060524, 0.020175, 0.010087, 0.0080699),
    phases=(1.3414, 1.3228, 2.9242, 2.8157, 2.6871, -2.8415),
)

Sternarchella_terminalis_harmonics = dict(
    amplitudes=(
        0.11457,
        0.4401,
        0.41055,
        0.20132,
        0.061364,
        0.011389,
        0.0057985,
    ),
    phases=(-2.7106, 2.4472, 1.6829, 0.79085, 0.119, -0.82355, -1.9956),
)

Sternopygus_dariensis_harmonics = dict(
    amplitudes=(
        0.98843,
        0.41228,
        0.047848,
        0.11048,
        0.022801,
        0.030706,
        0.019018,
    ),
    phases=(1.4153, 1.3141, 3.1062, -2.3961, -1.9524, 0.54321, 1.6844),
)

wavefish_harmonics = dict(
    Sine=Sine_harmonics,
    Alepto=Apteronotus_leptorhynchus_harmonics,
    Arostratus=Apteronotus_rostratus_harmonics,
    Eigenmannia=Eigenmannia_harmonics,
    Sternarchella=Sternarchella_terminalis_harmonics,
    Sternopygus=Sternopygus_dariensis_harmonics,
)


def wavefish_spectrum(
    fish: Union[str, dict, Tuple[List[float], List[float]]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the amplitudes and phases of a wavefish EOD.

    Parameters
    ----------
    fish : str, dict, or tuple of lists/arrays
        Specify relative amplitudes and phases of the fundamental and its harmonics.
        If string then take amplitudes and phases from the `wavefish_harmonics` dictionary.
        If dictionary then take amplitudes and phases from the 'amplitudes' and 'phases' keys.
        If tuple then the first element is the list of amplitudes and
        the second one the list of relative phases in radians.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Amplitudes and phases of the fundamental and its harmonics.

    Raises
    ------
    KeyError:
        Unknown fish.
    IndexError:
        Amplitudes and phases differ in length.
    """
    if isinstance(fish, (tuple, list)):
        amplitudes = np.array(fish[0])
        phases = np.array(fish[1])
    elif isinstance(fish, dict):
        amplitudes = np.array(fish["amplitudes"])
        phases = np.array(fish["phases"])
    else:
        if fish not in wavefish_harmonics:
            raise KeyError(
                "unknown wavefish. Choose one of " + ", ".join(wavefish_harmonics.keys())
            )
        amplitudes = np.array(wavefish_harmonics[fish]["amplitudes"])
        phases = np.array(wavefish_harmonics[fish]["phases"])

    if len(amplitudes) != len(phases):
        raise IndexError("need exactly as many phases as amplitudes")

    # remove NaNs:
    for k in reversed(range(len(amplitudes))):
        if np.isfinite(amplitudes[k]) or np.isfinite(phases[k]):
            amplitudes = amplitudes[: k + 1]
            phases = phases[: k + 1]
            break

    return amplitudes, phases


def wavefish_eods(
    fish: Union[str, Dict[str, Union[List[float], np.ndarray]]] = "Alepto",
    frequency: Union[float, np.ndarray] = 100.0,
    samplerate: float = 44100.0,
    duration: float = 1.0,
    phase0: float = 0.0,
    noise_std: float = 0.05,
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
    # get relative amplitude and phases:
    amplitudes, phases = wavefish_spectrum(fish)

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
    eod = wavefish_eods(
        "Alepto",
        frequency=100.0,
        samplerate=44100.0,
        duration=0.05,
    )
    plt.plot(eod)
    plt.show()


if __name__ == "__main__":
    main()
