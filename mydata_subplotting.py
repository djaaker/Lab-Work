import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import (make_axes_locatable, ImageGrid,
                                     inset_locator)

import mne

import os


subjects_dir = mne.datasets.sample.data_path() / 'subjects'
selected_patient = "Patient 1"
patient = selected_patient

all_patients = False

fnirs_data_folder = r'C:/Users/dalto/Downloads/Shlab/Shlab_data'

brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='low_contrast', hemi='both')
#defines brain as the command that activates the freeview mode

pat_data = []

# Use os.listdir() to get a list of all files and directories in the specified directory
folder_contents = os.listdir(fnirs_data_folder)

# Use a for loop to print the names of the directories within the specified directory
for item in folder_contents:
    if os.path.isdir(os.path.join(fnirs_data_folder, item)):
        print(item)
        pat_data.append(item)

if all_patients:
    # fig,axes = plt.subplots(nrows=patient_cnt, ncols=3, figsize=(10, 4 * patient_cnt))
    # brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='low_contrast', hemi='both')
    
    # for patient in pat_data:
    #     patient_data_folder = os.path.join(fnirs_data_folder, patient)
    #     raw_intensity = mne.io.read_raw_nirx(patient_data_folder, verbose=True)
    #     brain.add_sensors(
    #         raw_intensity.info, trans='fsaverage', meg= 'sensors',
    #         fnirs=['channels', 'pairs', 'sources', 'detectors'])
    #         # if you want to input the just sources in the figure swap this line for "  fnirs=['sources'])  "
    #         # "  fnirs=['channels', 'pairs', 'sources', 'detectors'])  " for the entire scaffold

    #     brain.subplot(cnt,0)
    #     cnt += 1
    #     print(patient)
    
    fig = plt.figure(figsize=(4, 4))
    axes = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.5)
    for ax, image, title in zip(axes, pat_data, pat_data):
        patient_data_folder = os.path.join(fnirs_data_folder, patient)
        raw_intensity = mne.io.read_raw_nirx(patient_data_folder, verbose=True)
        
        brain.add_sensors(
                        raw_intensity.info, trans='fsaverage', meg= 'sensors',
                        fnirs=['channels', 'pairs', 'sources', 'detectors'])

        ax.imshow(image)
        ax.set_title('{} cropping'.format(title))


else:
    patient_data_folder = os.path.join(fnirs_data_folder, patient)
    raw_intensity = mne.io.read_raw_nirx(patient_data_folder, verbose=True)
    brain.add_sensors(
        raw_intensity.info, trans='fsaverage', meg= 'sensors',
        fnirs=['channels', 'pairs', 'sources', 'detectors'])
        #if you want to input the just sources in the figure swap this line for "  fnirs=['sources'])  "
        # "  fnirs=['channels', 'pairs', 'sources', 'detectors'])  " for the entire scaffold

    # brain.show_view(azimuth=20, elevation=60, distance=400)
    #runs the freeview mode

    # print("\n\n" + str(all_patients) + ", patient count: " + str(patient_cnt) +  "\n\n")

# screenshot = brain.screenshot()
# brain.close()

# # plt.show()

# nonwhite_pix = (screenshot != 255).any(-1)
# nonwhite_row = nonwhite_pix.any(1)
# nonwhite_col = nonwhite_pix.any(0)
# cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

fig = plt.figure(figsize=(4, 4))
axes = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.5)

images = []
images = [None] * len(pat_data)  # initialize images list with correct length


for i, patient in enumerate(pat_data):
    brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='low_contrast', hemi='both')
    patient_data_folder = os.path.join(fnirs_data_folder, patient)
    raw_intensity = mne.io.read_raw_nirx(patient_data_folder, verbose=True)

    brain.add_sensors(
        raw_intensity.info, trans='fsaverage', meg= 'sensors',
        fnirs=['channels', 'pairs', 'sources', 'detectors'])

    brain.show_view(azimuth=20, elevation=60, distance=400)
    
    screenshot = brain.screenshot()
    brain.close()

    images[i] = screenshot

cnt = 1
for ax, image in zip(axes, images):
    ax.imshow(image)
    ax.set_title('Patient ' + str(cnt))
    cnt += 1

# for ax, patient in zip(axes, pat_data):
#     image = brain.screenshot
#     ax.imshow(image)


# for ax, image, title in zip(axes, [screenshot, cropped_screenshot],
#                             ['Before', 'After']):
#     ax.imshow(image)
#     ax.set_title('{} cropping'.format(title))
    
plt.show()









# plots = [None] * len(pat_data)  # initialize plots list with correct length
#         fig = plt.figure(figsize=(8, 8))
#         axes = ImageGrid(fig, 111, nrows_ncols=(math.ceil(len(pat_data) // 3), 3), axes_pad=0.5)

#         for i, patient in enumerate(pat_data):
#             patient_data_folder = os.path.join(fnirs_data_folder, patient)
#             sample_data_raw_path = patient_data_folder
#             sample_raw = mne.io.read_raw_nirx(
#                 sample_data_raw_path, preload=False, verbose=False)

#             #creates a matplot to show the 2D plot
#             fig = plt.figure()
#             ax3d = fig.add_subplot(122, projection='3d')
#             #uses mne library to use their built-in "plot_sensors" function and specify 3D fnirs amplitude data 
#             subplot = sample_raw.plot_sensors(ch_type='fnirs_cw_amplitude', axes=ax3d, kind='3d')
#             ax3d.view_init(azim=70, elev=15)

#             plots[i] = fig

#         cnt = 1
#         for ax, subplot in zip(axes, plots):
#             ax.plot(subplot)
#             ax.set_title('Patient ' + str(cnt))
#             cnt += 1





# for i, patient in enumerate(pat_data):
        #     row = i // n_cols
        #     col = i % n_cols
        #     ax = axs[row, col]
        #     ax.set_title(patient)
        #     patient_data_folder = os.path.join(fnirs_data_folder, patient)
        #     fnirs_raw_dir = patient_data_folder
        #     raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
            
        #     # raw_od.plot(n_channels=len(raw_od.ch_names), duration=500)
        #     # ax.set_xlabel(None)
        #     # ax.set_ylabel(None)

        #     raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
        #     raw_od.plot(n_channels=len(raw_od.ch_names), duration=500)
        #     ax = axs[row, col]
        #     ax.set_title(patient)

        # plt.show()