import math 
import numpy as np
import itertools
from scipy.signal import medfilt


def adjust_offset(peak, offset, wt_at_scale, sampling_rate, max_distance=0.08,shift=3): 
    new_off = zero_crossing(wt_at_scale,offset,len(wt_at_scale)-1)
    max_distance = int(math.ceil(max_distance*sampling_rate) + 1)
    if new_off :
        new_off  +=shift
    else:
        new_off  = offset  + shift
    if abs(peak - offset)> max_distance:
        new_off = peak + max_distance
    return new_off

def adjust_onset(peak, onset, wt_at_scale, sampling_rate, max_distance=0.08, shift=3):
    adjusted_on, reversed_wt1 = [], wt_at_scale[::-1]
    max_distance = int(math.ceil(max_distance*sampling_rate) + 1)
    reversed_onset = len(wt_at_scale) - onset
    adjusted_reversed_on = zero_crossing(reversed_wt1,reversed_onset,len(reversed_wt1)-1)
    if adjusted_reversed_on:
        new_on = len(wt_at_scale)-adjusted_reversed_on-shift
    else:
        new_on = onset - shift
    if abs(peak - new_on)> max_distance:
        new_on = peak - max_distance
    return new_on

def keep_qrs(rpeaks):
    intervals = rr_intervals(rpeaks)
    median_rr = np.median(intervals)
    idx_qrs = [index for index, value in enumerate(intervals) if value>=0.5*median_rr]
    return idx_qrs

def rr_intervals(rpeaks):
    return [abs(rpeak_curr - rpeak_nxt) for rpeak_curr, rpeak_nxt in zip(rpeaks[:-1],rpeaks[1:])]

def is_candidate_qrs(left, right, wt_scale_one, sampling_rate):
    window = int(math.ceil(125*sampling_rate/1000) + 1) 
    return abs(left - right)<=window and not is_same_sign(wt_scale_one[left], wt_scale_one[right]) 

def zero_crossing(signal, start, end):
    zero_pt, index = None, start
    try :
        while is_same_sign(signal[index], signal[index+1]) and index<=end:
            index += 1
        zero_pt = index 
    except IndexError:
        return None
    else:
        return zero_pt

def maximum_scale_one(maximum, wt_transforms, sampling_rate): 
    starting_scale = 3
    while starting_scale>0: 
        candidate_maxima = find_candidate_maxima(maximum, wt_transforms[starting_scale], wt_transforms[starting_scale-1],sampling_rate)
        if not candidate_maxima:
            return None
        else:
            best_maximum = choose_best_maximum(maximum, candidate_maxima)  
            maximum = best_maximum
            starting_scale -=1 
    return best_maximum

def choose_best_maximum(maximum, candidate_maxima):
    if len(candidate_maxima) == 1:
        return candidate_maxima[0]
    distance = [abs(candidate - maximum) for candidate in candidate_maxima] 
    return candidate_maxima[np.argmin(distance)] 

def find_candidate_maxima(maximum, wt_prev, wt_next, sampling_rate):
    window = int(math.ceil(40*sampling_rate/1000) + 1)
    start = max(0,maximum-window)
    end = min(maximum+window+1, len(wt_next)) 
    candidate_maxima = find_significant_maxima(abs(wt_next[start:end]), rms(wt_next)) 
    candidate_maxima = [candidate + start for candidate in candidate_maxima if is_same_sign(wt_next[candidate+start],wt_prev[maximum])]
    return candidate_maxima

def is_same_sign(val1,val2):
    return np.sign(val1) == np.sign(val2) 

def find_significant_maxima(signal, threshold):
    maxima = [index for index, value in enumerate(signal) if value>threshold and is_local_maximum(index, signal)] 
    return maxima

def is_local_maximum(index, signal): 
    if index<2 or index>len(signal)-3:
        return False
    curr = signal[index]
    prev1, prev2 = signal[index-1], signal[index-2]
    next1, next2 = signal[index+1], signal[index+2]
    return is_greater_than(curr, prev1, prev2) and is_greater_than(curr, next1, next2)

def is_greater_than(curr, val1, val2):
    return curr>=val1 and curr>val2

def rms(signal):
    return np.sqrt(np.mean(signal**2))

def flatten(lst):
    return list(itertools.chain.from_iterable(lst))

def dropnan(lst):
    return [x for x in lst if str(x) != 'nan']

def baseline_correction(signal):   
    baseline = medfilt(signal, 49)  
    baseline = medfilt(baseline, 149)  
    corrected_signal = signal - baseline 
    return corrected_signal 