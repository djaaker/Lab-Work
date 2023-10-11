import numpy as np
import mne
from itertools import compress
from mne import Epochs, events_from_annotations

from mne_bids import BIDSPath, read_raw_bids
from mne_nirs.signal_enhancement import (enhance_negative_correlation,
                                         short_channel_regression)
from mne.preprocessing.nirs import optical_density, beer_lambert_law, scalp_coupling_index, temporal_derivative_distribution_repair





from mne.preprocessing.nirs import optical_density
from itertools import compress

import pandas as pd
from itertools import compress
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

# Import MNE processing
import mne
from mne.viz import plot_compare_evokeds
from mne import Epochs, events_from_annotations, set_log_level

# Import MNE-NIRS processing
import mne_nirs
from mne_nirs.channels import get_long_channels, get_short_channels
from mne_nirs.channels import picks_pair_to_idx as p2idx
from mne_nirs.datasets import fnirs_motor_group
from mne.preprocessing.nirs import beer_lambert_law, optical_density,\
    temporal_derivative_distribution_repair, scalp_coupling_index
from mne_nirs.signal_enhancement import (enhance_negative_correlation,
                                         short_channel_regression)

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids

# Import StatsModels
import statsmodels.formula.api as smf

# Import Plotting Library
import matplotlib.pyplot as plt
import lets_plot
from lets_plot import *

# Imports libraries related to GLM Ananlysis
import h5py
from scipy import interpolate
import numpy as np
from mne_nirs.experimental_design import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from mne_nirs.statistics import run_glm
from mne import read_evokeds

from mne_bids import BIDSPath





# NORMALIZATION FUNCTION (for single event)
def normalize(raw_haemo,epochs):
    hold_data_all = epochs["1.0"].average() # take the average of breath holds for this subject
    hold_data_all = hold_data_all.get_data() # get the data

    max_during_hold = np.max(hold_data_all) # this should be a 1 x num_channels array

    z = raw_haemo.get_data() # extract (only long channel) data from oxy and deoxy-hemoglobin concentration matrix
    z = np.transpose(z) / max_during_hold # Divide all traces in each channel by the maximum change in concentration during the breath hold.

    # We are now in arbitrary units (deltaHb / deltaHb)
    raw_haemo._data= np.transpose(z) # replace the data in raw_haemo

    return raw_haemo, epochs


# ANALYSIS FUNCTION (for single subject)
def individual_analysis(bids_path):
    # Read data with annotations in BIDS format
    raw_intensity = read_raw_bids(bids_path, verbose=False)

    # Convert signal to optical density and determine bad channels
    raw_od = optical_density(raw_intensity)
    sci = scalp_coupling_index(raw_od, h_freq=1.35, h_trans_bandwidth=0.1)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))
    raw_od.interpolate_bads()

    # Downsample and apply signal cleaning techniques
    raw_od.resample(1)
    raw_od = temporal_derivative_distribution_repair(raw_od)
    raw_od = short_channel_regression(raw_od)

    # Convert to haemoglobin and filter
    raw_haemo = beer_lambert_law(raw_od)
    
    ## FILTERS OUT HEART RATE
    raw_haemo = raw_haemo.filter(None, 0.4,
                                 h_trans_bandwidth=0.1, l_trans_bandwidth=0.01,
                                 verbose=False)
    raw_haemo.annotations.delete(raw_haemo.annotations.description == '15')
    # Apply further data cleaning techniques and extract epochs
    raw_haemo = enhance_negative_correlation(raw_haemo)

    # Pick data channels that are actually informative
    roi_channels = mne.pick_channels(raw_haemo.info['ch_names'], include=['Left_PT','Right_PT'])
    raw_haemo = raw_haemo.copy().pick_channels(roi_channels)

    # Extract events but ignore those with
    events, event_dict = events_from_annotations(raw_haemo, verbose=False,
                                                 regexp='^(?![Ends]).*$')

    epochs = Epochs(raw_haemo, events, event_id=event_dict, tmin=-5, tmax=30,
                    reject=dict(hbo=100e-6), reject_by_annotation=True,
                    proj=True, baseline=(None, 0), detrend=1,
                    preload=True, verbose=False,event_repeated='merge')
    
    raw_haemo, epochs = normalize(raw_haemo,epochs)

    return raw_haemo, epochs



# MAIN FUNCTION
nsubs = 10   # input number of subjects
all_evokeds = defaultdict(list)

for sub in range(1,nsubs+1):
        subject_id = "%02d" % sub
        bids_path = BIDSPath(
            subject="%02d" % sub,
            task="task",
            # session='01',
            datatype="nirs",
            suffix='nirs',
            root=r"C:\Users\dalto\Downloads\project\sourcedata_lm",
            extension=".snirf"
        )

        raw_haemo, epochs = individual_analysis(bids_path)
        
        # get haemo channel epochs (hbo, hbr, and/or fnirs channels)
        epochs = epochs[["1.0", "2.0"]].pick('hbo')

        # Analyse data and return both ROI and channel results
        raw_haemo, epochs = individual_analysis(bids_path)

        # Save individual-evoked participant data along with others in all_evokeds
        for cidx, condition in enumerate(epochs.event_id):
            all_evokeds[condition].append(epochs[condition].average())


        roi_lists = {
            'Left_IFG' : [[1,1], [2,1], [3,1], [4,1], [3,2], [4,2]],
            'Left_HG' : [[5,3], [5,4], [5,5], [6,3], [6,4], [6,5]],
            'Left_PT' :[[6,6], [6,7], [7,5], [7,7],  [8,6], [8,7]],
            'Right_IFG' :  [[9,8], [10,8], [11,8], [11,9], [12,8], [12,9]],
            'Right_HG' : [[13,10], [13,11], [13,12], [14,10], [14,11], [14,12]],
            'Right_PT' : [[14,13], [14,14], [15,12], [15,14], [16,13], [16,14]]
        }


        # Assign the contents of each ROI list to separate variables
        Left_IFG = roi_lists['Left_IFG']
        Left_HG = roi_lists['Left_HG']
        Left_PT = roi_lists['Left_PT']
        Right_IFG = roi_lists['Right_IFG']
        Right_HG = roi_lists['Right_HG']
        Right_PT = roi_lists['Right_PT']

        # Name ROIs and pair each list to the raw type haemo data
        rois = dict(Left_IFG=p2idx(raw_haemo, [[1,1], [2,1], [3,1], [4,1], [3,2], [4,2]],
                            on_missing='warning'),
                    Left_HG=p2idx(raw_haemo, [[5,3], [5,4], [5,5], [6,3], [6,4], [6,5]],
                                on_missing='warning'),

                    Left_PT=p2idx(raw_haemo, [[6,6], [6,7], [7,5], [7,7],  [8,6], [8,7]],
                                on_missing='warning'),
                    Right_HG=p2idx(raw_haemo, [[13,10], [13,11], [13,12], [14,10], [14,11], [14,12]],
                                on_missing='warning'),
                    Right_PT=p2idx(raw_haemo, [[14,13], [14,14], [15,12], [15,14], [16,13], [16,14]],
                                on_missing='warning'),
                    Right_IFG=p2idx(raw_haemo, [[9,8], [10,8], [11,8], [11,9], [12,8], [12,9]],
                                    on_missing='warning'))

        df = pd.DataFrame(columns=['ID', 'ROI', 'Chroma', 'Condition', 'Value']) #creates pandas dataframe to hold specified data
        # PRINTS AMPLITUDES OF EVOKED RESPONSES FOR EACH TRIGGER AT EACH ROI
        for idx, evoked in enumerate(all_evokeds):
            for id, subj_data in enumerate(all_evokeds[evoked]):
                for roi in rois:
                    for chroma in ["hbo", "hbr"]:
                        # Get subject ID
                        subj_id = id

                        # Extract data for the specified ROI and chroma
                        data = deepcopy(subj_data).pick(picks=rois[roi]).pick(chroma)

                        # Calculate the mean value of the data within a specified time range
                        value = data.crop(tmin=3.0, tmax=8.0).data.mean() * 1.0e6

                        # Append metadata and extracted feature to the dataframe
                        new = {'ID': subj_id, 'ROI': roi, 'Chroma': chroma, 'Condition': evoked, 'Value': value}
                        new_df = pd.DataFrame(new, index=[0])
                        df = pd.concat([df, new_df])

        for pick, color in zip(['hbr','hbo'],['r','b']):
            plot_compare_evokeds({evoked:all_evokeds['1.0']},combine='mean',picks=pick,colors=[color])
        plt.show()