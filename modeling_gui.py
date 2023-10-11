import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog

from PIL import Image,ImageTk

import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

import mne
from mne.datasets import sample

from mpl_toolkits.mplot3d import Axes3D 
from pathlib import Path
import mne
import math
import scipy.io as sio
import pandas as pd

import os
import pickle

import pyvista as pv

from mpl_toolkits.axes_grid1 import (make_axes_locatable, ImageGrid,
                                     inset_locator)

# glm_est = run_glm(raw_heamo,design_matrix)


## Initialization of the program-wide variables 
subjects_dir = mne.datasets.sample.data_path() / 'subjects'
selected_patient = "Patient 1"


## BELOW CODE USES THE PICKLE LIBRARY TO STORE THE USER'S NIRS DATA FOLDER LOCATION
# Check if the path to the saved directory exists
path_to_pickle_file = "fnirs_data_folder.pkl"

if os.path.isfile(path_to_pickle_file):
    # Load the path from the saved pickle file
    with open(path_to_pickle_file, "rb") as f:
        fnirs_data_folder = pickle.load(f)

else:
    # Ask the user to select a directory
    fnirs_data_folder = filedialog.askdirectory(title="Please select the folder containing NIRS data")

    # Save the selected directory to a pickle file
    with open(path_to_pickle_file, "wb") as f:
        pickle.dump(fnirs_data_folder, f)


def num_patients():
    # Specify the path of the folder you want to check
    folder_path = fnirs_data_folder

    # Get the list of all files and folders in the specified folder
    folder_contents = os.listdir(folder_path)

    # Count the number of folders in the specified folder
    folder_count = 0
    for item in folder_contents:
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_count += 1

    return folder_count

# creates the popup that asks which patient data to use
def popup():
    # Uses user input to choose which patient info to show
    def choose_patient():
        # Get the selected patient from the dropdown list
        global selected_patient
        selected_patient = patient_dropdown.get()
        global all_patients
        if selected_patient == "All Patients (WIP)":
            all_patients = True
        else:
            all_patients = False

        popup_window.destroy()

    popup_window = tk.Tk()
    popup_window.title("Select Patient Data")
    # Create a label to prompt the user to select a patient
    label = ttk.Label(popup_window, text="Select a patient:")
    label.pack(padx=10, pady=10)

    # Create a dropdown list with the available patients
    patients = ["All Patients (WIP)"]
    patients.extend([f"Patient {i}" for i in range(1, num_patients() + 1)])
    patient_dropdown = ttk.Combobox(popup_window, values=patients)
    patient_dropdown.current(0)
    patient_dropdown.pack(padx=10, pady=10)


    # Create a button to submit the selected patient
    submit_button = ttk.Button(popup_window, text="Submit", command=choose_patient)
    submit_button.pack(padx=10, pady=10)

    popup_window.wait_window()

# CREATES PREOPENING WINDOW FOR PATIENT DATA
popup()

patient_cnt = num_patients()
print('\n\n' + str(patient_cnt) + '\n\n')
patient = selected_patient
if selected_patient != "All Patients (WIP)":
    # print(patient)
    patient_data_folder = os.path.join(fnirs_data_folder, patient)
    fnirs_cw_amplitude_dir = patient_data_folder
    # print("\n\n" + fnirs_data_folder + "\n\n")
    raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)

# Initialize an empty list to store the names of the directories
pat_data = []

# Use os.listdir() to get a list of all files and directories in the specified directory
folder_contents = os.listdir(fnirs_data_folder)

# Use a for loop to print the names of the directories within the specified directory
for item in folder_contents:
    if os.path.isdir(os.path.join(fnirs_data_folder, item)):
        print(item)
        pat_data.append(item)
        
# Print the list of directories
print(pat_data)

# CREATES MAIN WINDOW
root = tk.Tk()
root.title("Shader Lab Brain Model GUI")

menu = tk.Menu(root)
item = tk.Menu(menu)

root.config(menu=menu)

# resizes an image to fit within a bounding box of size width x height.
def format_image(image, size):
    # Convert PhotoImage to PIL Image
    pil_image = ImageTk.getimage(image)
    # Get original dimensions
    width, height = pil_image.size
    # Crop the image to a square with the shorter side unchanged
    if width < height:
        left = 0
        top = (height - width) // 2
        right = width
        bottom = top + width
    else:
        left = (width - height) // 2
        top = 0
        right = left + height
        bottom = height
    cropped_image = pil_image.crop((left, top, right, bottom))
    # Resize the image to the desired size
    resized_image = cropped_image.resize((size, size), Image.LANCZOS)
    # Convert PIL Image to PhotoImage
    resized_photoimage = ImageTk.PhotoImage(resized_image)
    return resized_photoimage


def pat_change():
    popup()
    pass


# creates the 3D scaffold showing the channels, pairs, sources, and detectors 
def scaff():
    global patient
    # brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='low_contrast', hemi='both')
    #defines brain as the command that activates the freeview mode

    if all_patients:
        # fig,axes = plt.subplots(nrows=patient_cnt, ncols=3, figsize=(10, 4 * patient_cnt))
        # brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='low_contrast', hemi='both')
        fig = plt.figure(figsize=(12, 12))
        axes = ImageGrid(fig, 111, nrows_ncols=(math.ceil(len(pat_data) // 4), 4), axes_pad=0.5)

        images = []
        images = [None] * len(pat_data)  # initialize images list with correct length


        for i, patient in enumerate(pat_data):
            brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='low_contrast', hemi='both')
            patient_data_folder = os.path.join(fnirs_data_folder, patient)
            raw_intensity = mne.io.read_raw_nirx(patient_data_folder, verbose=True)

            brain.add_sensors(
                raw_intensity.info, trans='fsaverage', meg= 'sensors',
                fnirs=['sources','detectors'])

            brain.show_view(azimuth=20, elevation=60, distance=500)
            
            screenshot = brain.screenshot()
            brain.close()

            images[i] = screenshot

        cnt = 1
        for ax, image in zip(axes, images):
            ax.imshow(image)
            ax.set_title('Patient ' + str(cnt))
            cnt += 1  

        plt.show()

    else:
        brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='low_contrast', hemi='both')
        # glm_est.copy().surface_projection(condition="Speech",view='dorsal', chroma="hbo", subjects_dir = subjects_dir)
        patient_data_folder = os.path.join(fnirs_data_folder, patient)
        raw_intensity = mne.io.read_raw_nirx(patient_data_folder, verbose=True)
        brain.add_sensors(
            raw_intensity.info, trans='fsaverage', meg= 'sensors',
            fnirs=['sources','detectors'])
        #if you want to input the just sources in the figure swap this line for "  fnirs=['sources'])  "
        # "  fnirs=['channels', 'pairs', 'sources', 'detectors'])  " for the entire scaffold

        brain.show_view(azimuth=20, elevation=60, distance=400)
        #runs the freeview mode

        plt.show()

    # print("\n\n" + str(all_patients) + ", patient count: " + str(patient_cnt) +  "\n\n")


# funciton to plot the 3D sensor points for the data previously selected
def threed():
    global patient
        
    if all_patients:
        n_patients = len(pat_data)
        n_cols = 3
        n_rows = math.ceil(n_patients / n_cols)
        fig, axs = plt.subplots(n_rows, n_cols, subplot_kw={'projection': '3d'},
                                figsize=(12, 4*n_rows))

        for i, patient in enumerate(pat_data):
            row = i // n_cols
            col = i % n_cols
            ax3d = axs[row, col]
            patient_data_folder = os.path.join(fnirs_data_folder, patient)
            sample_data_raw_path = patient_data_folder
            sample_raw = mne.io.read_raw_nirx(
                sample_data_raw_path, preload=False, verbose=False)

            # uses mne library to use their built-in "plot_sensors" function and specify 3D fnirs amplitude data 
            sample_raw.plot_sensors(ch_type='fnirs_cw_amplitude', axes=ax3d, kind='3d')
            ax3d.view_init(azim=70, elev=15)
            ax3d.set_title(patient)

        plt.show()

    else:
        patient_data_folder = os.path.join(fnirs_data_folder, patient)
        sample_data_raw_path = patient_data_folder
        sample_raw = mne.io.read_raw_nirx(
            sample_data_raw_path, preload=False, verbose=False)

        #creates a matplot to show the 2D plot
        fig = plt.figure()
        ax3d = fig.add_subplot(122, projection='3d')
        #uses mne library to use their built-in "plot_sensors" function and specify 3D fnirs amplitude data 
        sample_raw.plot_sensors(ch_type='fnirs_cw_amplitude', axes=ax3d, kind='3d')
        ax3d.view_init(azim=70, elev=15)

        plt.show()

# function to plot the 2D data for the data previously selected
def twod():
    global patient
        
    if all_patients:
        num_plots = len(pat_data)
        num_cols = 3
        num_rows = (num_plots - 1) // num_cols + 1  # calculate number of rows needed
        fig_width = num_cols * 5  # set width of figure based on number of columns
        fig_height = num_rows * 5  # set height of figure based on number of rows
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
        # num_rows = (len(pat_data) // 3) + (len(pat_data) % 3 > 0)
        # fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))

        for i, patient in enumerate(pat_data):
            patient_data_folder = os.path.join(fnirs_data_folder, patient)
            sample_data_raw_path = patient_data_folder
            sample_raw = mne.io.read_raw_nirx(sample_data_raw_path, preload=False, verbose=False)

            row = i // 3
            col = i % 3

            ax = axs[row, col]
            ax.set_title(patient)
            sample_raw.plot_sensors(ch_type='fnirs_cw_amplitude', axes=ax)

        plt.tight_layout()
        plt.show()

    else:
        patient_data_folder = os.path.join(fnirs_data_folder, patient)
        sample_data_raw_path = patient_data_folder
        sample_raw = mne.io.read_raw_nirx(sample_data_raw_path, preload=False, verbose=False)

        #creates a matplot to show the 2D plot
        fig = plt.figure()
        ax2d = fig.add_subplot(121)
        #uses mne library to use their built-in "plot_sensors" function and specify 2D fnirs amplitude data 
        sample_raw.plot_sensors(ch_type='fnirs_cw_amplitude', axes=ax2d)

        plt.show()


# graphs the optical density for the data
def od():
    global patient
        
    if all_patients:
        n_patients = len(pat_data)
        n_cols = 3
        n_rows = math.ceil(n_patients / n_cols)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 5), constrained_layout=True)
        # plots = [None] * len(pat_data)

        fig = plt.figure()
        
        for i, patient in enumerate(pat_data):
            patient_data_folder = os.path.join(fnirs_data_folder, patient)
            fnirs_raw_dir = patient_data_folder
            raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()\

            raw_od = fig.add_subplot()
            print(mne.preprocessing.nirs.optical_density(raw_intensity))

            row = i // 3
            col = i % 3

            ax = axs[row, col]
            ax.set_title(patient)
            raw_od.plot(n_channels=len(raw_od.ch_names), duration=500)

        plt.tight_layout()
        plt.show()

    else:
        patient_data_folder = os.path.join(fnirs_data_folder, patient)
        fnirs_raw_dir = patient_data_folder
        raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()

        raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
        raw_od.plot(n_channels=len(raw_od.ch_names), duration=500)
        plt.show()


def haemo():
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=500)

    fig = raw_haemo.plot_psd(average=True)
    fig.suptitle('Before filtering', weight='bold', size='x-large')
    fig.subplots_adjust(top=0.88)
    raw_haemo = raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                                l_trans_bandwidth=0.02)
    fig = raw_haemo.plot_psd(average=True)
    fig.suptitle('After filtering', weight='bold', size='x-large')
    fig.subplots_adjust(top=0.88)

    plt.show()


path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

image_library = dir_path + '\\bmg_photos'

# print("Current running .py file location:", path)
# print("Current running .py file directory location:", dir_path)
# print(image_library)

size = 250

# chooses an image to be placed on each button 
scaff_image = ImageTk.PhotoImage(file=image_library + '\\Scaffold.png')
# crops and resize image to be a square
scaff_image = format_image(scaff_image,size)

threed_image = tk.PhotoImage(file=image_library + '\\threed_image.png')
threed_image = format_image(threed_image,size)

twod_image = tk.PhotoImage(file=image_library + '\\twod_image.png')
twod_image = format_image(twod_image,size)

od_image = tk.PhotoImage(file=image_library + '\\od_image.png')
od_image = format_image(od_image,size)

haemo_image = tk.PhotoImage(file=image_library + '\\haemo_image.png')
haemo_image = format_image(haemo_image,size)


scaff_button = tk.Button(root, image=scaff_image,command=lambda:[scaff()])
scaff_button.grid(row=0, column=0, padx=10, pady=10)
scaff_button.config(text='NIRS Scaffold', compound="top")

threed_button = tk.Button(root, text='3D Plot',image=threed_image,command=lambda:[threed()])
threed_button.grid(row=0, column=1, padx=10, pady=10)
threed_button.config(text='3D Plot', compound="top")

twod_button = tk.Button(root, text='2D Plot',image=twod_image,command=lambda:[twod()])
twod_button.grid(row=0, column=2, padx=10, pady=10)
twod_button.config(text='2D Plot', compound="top")

od_button = tk.Button(root, text='Optical Density',image=od_image,command=lambda:[od()])
od_button.grid(row=1, column=0, padx=10, pady=10)
od_button.config(text='Optical Density', compound="top")

haemo_button = tk.Button(root, text='Haemoglobin Concentrations',image=haemo_image,command=lambda:[haemo()])
haemo_button.grid(row=1, column=1, padx=10, pady=10)
haemo_button.config(text='Haemoglobin Concentrations', compound="top")

button6 = tk.Button(root, text='Button 6')
button6.grid(row=1, column=2, padx=10, pady=10)

change_button = tk.Button(root, text= "Change Patient",command= pat_change)
change_button.place(x=0, y=0)


root.mainloop()