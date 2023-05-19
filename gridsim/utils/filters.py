import numpy as np
from scipy.signal import butter, sosfiltfilt


def lowpass_filter(
    signal: np.ndarray, samplerate: float, cutoff: float
) -> np.ndarray:
    """Lowpass filter a signal.

    Parameters
    ----------
    data : np.ndarray
        The data to be filtered
    rate : float
        The sampling rate
    cutoff : float
        The cutoff frequency

    Returns
    -------
    np.ndarray
        The filtered data
    """
    sos = butter(2, cutoff, "lowpass", fs=samplerate, output="sos")
    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal
