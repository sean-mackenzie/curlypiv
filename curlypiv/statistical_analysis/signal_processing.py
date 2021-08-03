import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal


def generate_waveform(function='sine', frequency=100, length=10, sampling_rate=20, amplitude=1, theta=0):
    """
    Generates a sine or square wave for signal convolution.
    Inputs:
        function (): 'sine' or 'square'
        frequency (Hz): frequency of the returned waveform.
        length (#): number of samples in the returned waveform.
        sampling_rate (Hz): sampling frequency of the returned waveform.
        amplitude (): always return an amplitude of 1 that can be scaled for each application.
        theta (rad): specify initial value of signal using theta.
    Returns:

    """
    total_time = length / sampling_rate

    if sampling_rate * 2 < frequency:
        print("Sampling rate needs to be >= Nyquist sampling rate. Changing sampling rate to Nyquist frequency.")
        sampling_rate = frequency * 2

    time = np.arange(start=0, stop=total_time, step=1/sampling_rate, dtype=float)

    if function == 'sine':
        raw_waveform = np.sin(2 * np.pi * frequency * time + theta)
        waveform = raw_waveform / np.max(raw_waveform) * amplitude
    elif function == 'square':
        waveform = signal.square(2 * np.pi *  frequency * time)
    else:
        raise ValueError("{} is not implemented yet. Choose 'sine' or 'square'".format(function))

    generated_signal = np.stack((time, waveform), axis=0)

    return generated_signal