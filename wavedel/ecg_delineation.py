import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import convolve as conv
from scipy.signal import resample, find_peaks 
from . import utils as ut
from . import wavelet as wvlt 

################################ Detect all fiducials ################################
def ecg_delineator(ecg, sampling_rate, adjust_rpeak=True):
    markers = {}
    qrs_peaks, qrs_on, qrs_off = detect_QRS(ecg, sampling_rate,adjust_rpeak)
    p_peaks, p_on, p_off = detect_pwave(ecg, qrs_on, sampling_rate)
    t_peaks, t_on, t_off = detect_twave(ecg, qrs_off, sampling_rate)
    markers['ECG_R_Peaks'] = qrs_peaks
    markers['ECG_R_Onsets'] = qrs_on
    markers['ECG_R_Offsets'] = qrs_off
    markers['ECG_P_Peaks'] = p_peaks
    markers['ECG_P_Onsets'] = p_on
    markers['ECG_P_Offsets'] = p_off
    markers['ECG_T_Peaks'] = t_peaks
    markers['ECG_T_Onsets'] = t_on
    markers['ECG_T_Offsets'] = t_off
    return markers

################################ Detect QRS complex ################################

def clean_detection(func):
    def inner(ecg, sampling_rate, adjust_rpeak):
        wt =  wvlt.wavelet_decomposition(ecg, len(ecg),sampling_rate)
        qrs_peaks,qrs_on,qrs_off =  func(ecg, sampling_rate,adjust_rpeak)  
        qrs_peaks,qrs_on,qrs_off = remove_redundancy(ecg, qrs_peaks,qrs_on,qrs_off,sampling_rate) #wt[2]  
        qrs_on = adjust_all_onsets(qrs_peaks, qrs_on, wt[0],sampling_rate)
        qrs_off = adjust_all_offsets(qrs_peaks, qrs_off, wt[0],sampling_rate)
        if adjust_rpeak:
            qrs_peaks,qrs_on,qrs_off = adjust_all_peaks(ecg, qrs_peaks, qrs_on, qrs_off)
        return qrs_peaks,qrs_on,qrs_off
    return inner

@clean_detection
def detect_QRS(ecg, sampling_rate, adjust_rpeak):
    wt =  wvlt.wavelet_decomposition(ecg, len(ecg),sampling_rate)
    max_scale_four = ut.find_significant_maxima(abs(wt[3]), 0.5*ut.rms(wt[3]))  
    max_scale_one = [ut.maximum_scale_one(maximum, wt, sampling_rate) for maximum in max_scale_four]
    max_scale_one = [maximum for maximum in max_scale_one if maximum]
    wt_scale_one = wt[0]
    qrs_peaks, qrs_on, qrs_off = [], [], [] 
    for left, right in zip(max_scale_one[:-1], max_scale_one[1:]):  
        if ut.is_candidate_qrs(left, right, wt_scale_one, sampling_rate):
            zero_pt = ut.zero_crossing(wt_scale_one,left,right) 
            if zero_pt : 
                if abs(ecg[zero_pt+1])>abs(ecg[left]) or abs(ecg[zero_pt+1])>abs(ecg[right]):
                    qrs_peaks.append(zero_pt+1)
                    qrs_on.append(left)
                    qrs_off.append(right)
                    continue
                
                if abs(ecg[zero_pt+1])<abs(ecg[left]) and len(ecg[left-50:left+50])>0:
                    peak = np.argmax(np.abs(ecg[left-50:left+50])) + left-50
                    qrs_peaks.append(peak)
                    qrs_on.append(peak-10)
                    qrs_off.append(peak+10) 
                    continue
                
                if abs(ecg[zero_pt+1])<abs(ecg[right]) and len(ecg[right-50:right+50])>0:
                    peak = np.argmax(np.abs(ecg[right-50:right+50])) + right-50
                    qrs_peaks.append(peak)
                    qrs_on.append(peak-10)
                    qrs_off.append(peak+10)   
                    
    return qrs_peaks, qrs_on, qrs_off 

def adjust_all_peaks(ecg, qrs_peaks, qrs_on, qrs_off): 
    new_qrs_peaks, new_qrs_on, new_qrs_off = [], [], []
    for left, peak, right in zip( qrs_on, qrs_peaks, qrs_off):  
        right = min(right, len(ecg)-1) 
        if abs(ecg[peak])<abs(ecg[left]) and len(ecg[left-50:left+50])>0:
            new_peak = np.argmax(np.abs(ecg[left-50:left+50])) + left-50
            new_qrs_peaks.append(new_peak)
            new_qrs_on.append(new_peak-10)
            new_qrs_off.append(new_peak+10) 
            continue
        if abs(ecg[peak])<abs(ecg[right]) and len(ecg[right-50:right+50])>0:
            new_peak = np.argmax(np.abs(ecg[right-50:right+50])) + right-50
            new_qrs_peaks.append(new_peak)
            new_qrs_on.append(new_peak-10)
            new_qrs_off.append(new_peak+10)   
            continue
        new_qrs_peaks.append(peak)
        new_qrs_on.append(left)
        new_qrs_off.append(right) 
    return new_qrs_peaks, new_qrs_on, new_qrs_off 

def remove_redundancy(ecg, qrs_peaks, qrs_on, qrs_off,sampling_rate):
    incorrect_qrs = []
    tolerance = int(math.ceil(300*sampling_rate/1000) + 1)
    qrs = list(zip(qrs_peaks, qrs_on, qrs_off))
    try:
        for index, value in enumerate(qrs_peaks):
            curr_peak, nxt_peak = value, qrs_peaks[index+1]   
            if abs(curr_peak-nxt_peak)< tolerance: 
                if abs(ecg[qrs_on[index]])<abs(ecg[qrs_on[index+1]]):
                    incorrect_qrs.append(index+1) 
                    continue
                if abs(ecg[curr_peak])<0.5*abs(ecg[nxt_peak]):
                    incorrect_qrs.append(index) 
                    continue 
                incorrect_qrs.append(index+1) 
    except IndexError:
        pass 
    incorrect_qrs = list(set(incorrect_qrs))
    if len(incorrect_qrs)==len(qrs):
        return [], [], []
    qrs_to_keep = [value for index, value in enumerate(qrs) if index not in incorrect_qrs]
    qrs_peaks, qrs_on, qrs_off = zip(*qrs_to_keep) 
    return list(qrs_peaks), list(qrs_on), list(qrs_off)

def adjust_all_onsets(qrs_peaks, qrs_on, wt_scale_one, sampling_rate):
    new_onsets = [ut.adjust_onset(peak, onset, wt_scale_one, sampling_rate) for peak, onset in zip(qrs_peaks, qrs_on)]
    return new_onsets

def adjust_all_offsets(qrs_peaks, qrs_off, wt_scale_one, sampling_rate):
    new_offsets = [ut.adjust_offset(peak, offset, wt_scale_one, sampling_rate) for peak, offset in zip(qrs_peaks, qrs_off)]
    return new_offsets

################################ Detect P wave ################################
def detect_pwave(ecg, qrs_on, sampling_rate):
    neigh = int(math.ceil(200*sampling_rate/1000) + 1)
    wave_limit = int(math.ceil(80*sampling_rate/1000) + 1)
    #_,qrs_on,_ = detect_QRS(ecg, sampling_rate)
    p_peaks = []
    p_on = []
    p_off = []
    for qrs_onset in qrs_on:
        lower_bound = qrs_onset-neigh
        upper_bound = qrs_onset - 20
        try:
            signal = ecg[lower_bound:upper_bound]
            peak =  np.argmax(abs(signal)) + lower_bound
        except (ValueError, IndexError):
            pass
        else:
            try:
                on = np.argmin(abs(ecg[peak - wave_limit:peak-25])) + peak - wave_limit
            except ValueError:
                on = peak - wave_limit
            try:
                off = np.argmin(abs(ecg[peak+25:peak + wave_limit])) + peak + 25
            except ValueError:
                off = peak + wave_limit
            p_peaks.append(peak)
            p_on.append(on)
            p_off.append(off)
    return  p_peaks, p_on, p_off

################################ Detect T wave ################################
def detect_twave(ecg, qrs_off, sampling_rate):
    ecg = ut.baseline_correction(ecg)
    wt = wvlt.wavelet_decomposition(ecg, len(ecg),sampling_rate)
    neigh = int(math.ceil(400*sampling_rate/1000) + 1)
    wave_limit = int(math.ceil(80*sampling_rate/1000) + 1)
    t_peaks, t_on, t_off  = [], [], []
    for qrs_offset in qrs_off:
        lower_bound = qrs_offset + 20
        upper_bound = qrs_offset + neigh
        try:
            signal = ecg[lower_bound:upper_bound]
            peak =  np.argmax(abs(signal)) + lower_bound   

        except (ValueError, IndexError):
            pass
        else:
            try:
                on = np.argmin(abs(ecg[peak - wave_limit:peak-25])) + peak - wave_limit
            except ValueError:
                on = peak - wave_limit
            try:
                off = np.argmin(abs(wt[3][peak+25:peak + wave_limit])) + peak + 25
            except ValueError:
                off = peak + wave_limit
            t_peaks.append(peak)
            t_on.append(on)
            t_off.append(off)
    return  t_peaks, t_on, t_off 

