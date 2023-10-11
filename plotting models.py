from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa

import mne

# builtin_montages = mne.channels.get_builtin_montages(descriptions=True)
# for montage_name, montage_description in builtin_montages:
#     print(f'{montage_name}: {montage_description}')

easycap_montage = mne.channels.make_standard_montage('easycap-M1')
# print(easycap_montage)

# easycap_montage.plot()  # 2D
# fig = easycap_montage.plot(kind='3d', show=False)  # 3D
# fig = fig.gca().view_init(azim=70, elev=15)  # set view angle for tutorial


fname = r"C:\Users\dalto\Downloads\project\sourcedata_lm\sub-10\ses-01\nirs\sub-10_ses-01_task-wings_nirs.snirf.snirf"
sample_raw = mne.io.read_raw_snirf(fname, verbose=False)

fig = plt.figure()
ax2d = fig.add_subplot(121)
# ax3d = fig.add_subplot(122, projection='3d')
# sample_raw.plot_sensors(ch_type='fnirs_cw_amplitude', axes=ax2d)
sample_raw.plot_sensors(ch_type='fnirs_cw_amplitude')

# sample_raw.plot_sensors(ch_type='fnirs_cw_amplitude', axes=ax3d, kind='3d')
# ax3d.view_init(azim=70, elev=15)

plt.show()