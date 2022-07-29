from wavedel.ecg_delineation import ecg_delineator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualise_detection(ecg, markers, sampling_rate = 500, start_sec=0, stop_sec=3.2, fontsize=15): 
    time = [convert_from_samples_to_sec(i, sampling_rate=sampling_rate) for i in range(len(ecg))]
    colors = {'R': 'y', 'P': 'g', 'T': 'r'}
    plt.figure(figsize=(15, 4), dpi=400)
    plt.plot(time, ecg, color = 'k', linewidth=0.8)  
    for key, item in markers.items():
        for marker in item:
            marker_time = convert_from_samples_to_sec(marker, sampling_rate=sampling_rate)
            marker_ampli = ecg[marker]
            plt.plot(marker_time, marker_ampli, 'o', color=colors[key.split('_')[1]], markersize=6) 
    plt.xlim(start_sec, stop_sec)
    plt.xlabel('Time (s)', fontsize=fontsize)
    plt.ylabel('Amplitude', fontsize=fontsize)
    plt.xticks(fontsize=fontsize); plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('example/delineation.png')

def convert_from_samples_to_sec(nb_samples, sampling_rate, milli=False):
    if milli:
        return 1000*nb_samples/sampling_rate 
    else:
        return nb_samples/sampling_rate 

if __name__ == "__main__": 
    #Deineate 10s single-lead ECG sampled at 500 Hz
    ecg = np.array(pd.read_csv('example/ecg.csv')['ecg']) 
    markers = ecg_delineator(ecg, sampling_rate = 
    500) 
    visualise_detection(ecg, markers)