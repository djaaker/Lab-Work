import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

import mne
from mne.datasets import sample
from mne.viz import ClickableImage  # noqa: F401
from mne.viz import (plot_alignment, snapshot_brain_montage, set_3d_view)
from mne.io.fiff.raw import read_raw_fif

import time

subjects_dir = mne.datasets.sample.data_path() / 'subjects'
fnirs_data_folder = r"C:\Users\dalto\OneDrive\Documents\Shlab\LM_compilation"
nirx_data = fnirs_data_folder


## 3D MODEL OF ALL POINTS


# brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='low_contrast', hemi='both')
raw_intensity = mne.io.read_raw_nirx(nirx_data, verbose=True)
# brain.add_sensors(
#     raw_intensity.info, trans='fsaverage', meg= 'sensors',
#     fnirs=['sources'])

# brain.show_view(azimuth=20, elevation=60, distance=400)
# #runs the freeview mode




raw = nirx_data + '\2022-10-03_001_probeInfo'

montage = raw.get_montage()

# mne.viz.plot_sensors(info,kind='select')


plt.show()

input('Pause')
