'''Adapted from https://github.com/caledezma/WTdelineator'''

import numpy as np
from scipy.signal import resample

def wavelet_decomposition(ecg, number_samples, sampling_rate):
    '''
    w = wavelet_decomposition(ecg, number_samples, sampling_rate)
    Performs the wavelet decomposition of a signal using the algorithme-a-trous.
    Inputs:
        ecg (numpy array): contains the signal to be decomposed.
        number_samples (int): the number of samples of the signal that will be decomposed.
        sampling_rate (float): the sampling frequency of the signal that will be decomposed.
    Output:
        w (list): numpy arrays [w1, w2, w3, w4, w5] containing the wavelet
        decomposition of the signal at scales 2^1..2^5.
    '''

    w = []
    Q = wavelet_filters(number_samples, sampling_rate)

    # Apply the filters in the frequency domain and return the result in the time domain
    for q in Q:
        w += [np.real(np.fft.ifft(np.fft.fft(ecg) * q))]

    return w

def wavelet_filters(number_samples, sampling_rate):
    '''
    Q = wavelet_filters(number_samples, sampling_rate)
    Creates the filters required to make the wavelet decomposition using the
    algorithme-a-trous. This routine first creates the filters at 250 Hz and
    resamples them to the required sampling frequency.
    Inputs:
        number_samples (int): the number of samples of the signal that will be decomposed.
        sampling_rate (float): the sampling frequency of the signal that will be decomposed.
    Output:
        Q (list): contains five numpy arrays [Q1, Q2, Q3, Q4, Q5] that are the
        five filters required to make the wavelet decomposition.
    '''

    # M is the number of samples at 250 Hz that will produce number_samples samples after
    # re-sampling the filters
    M =  number_samples* 250/sampling_rate
    w = np.arange(0,2*np.pi, 2*np.pi/M) # Frequency axis in radians

    nscales = 5
    # Construct the filters at 250 Hz as specified in the paper
    Q = [_high_pass(w)]
    for k in range(2,nscales+1):
        G = _low_pass(w)
        for l in range(1,k-1):
            G *= _low_pass(2**l * w)
        Q += [_high_pass(2 ** (k-1) * w) * G]

    # Resample the filters from 250 Hz to the desired sampling frequency
    for i in range(len(Q)):
        Q[i] = np.fft.fft(resample(np.fft.ifft(Q[i]),number_samples))

    return Q


def _low_pass(w):
    '''
    H = low_pass(w)
    Constructs the low-pass filters required for the wavelet-based ECG delineator
    at a sampling frequency of 250 Hz.
    Input:
        w (numpy array): contains the frequency points, in radians, that
        will be used to construct the filter. w must be between 0 and 2pi.
    Output:
        H (numpoy array): contains the function H(w) = exp(1j*w/2) * cos(w/2)**3.
    '''
    return np.exp(1j*w/2) * np.cos(w/2) ** 3


def _high_pass(w):
    '''
    G = high_pass(w)
    Constructs the high-pass filters required for the wavelet-based ECG delineator
    at a sampling frequency of 250 Hz.
    Input:
        w (numpy array): contains the frequency points, in radians, that
        will be used to construct the filter. w must be between 0 and 2pi.
    Output:
        G (numpy array): contains the function H(w) = exp(1j*w/2) * cos(w/2)**3.
    '''

    return 4j*np.exp(1j*w/2)*np.sin(w/2)