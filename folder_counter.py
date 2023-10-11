import os

import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog

import mne

fnirs_data_folder = filedialog.askdirectory()

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

def choose_patient():
    # Get the selected patient from the dropdown list
    selected_patient = patient_dropdown.get()

    # If the user selected "All Patients", show all patient data
    if selected_patient == "All Patients":
        fnirs_cw_amplitude_dir = fnirs_data_folder
    else:
        patient_dir = os.path.join(fnirs_data_folder, selected_patient)
        fnirs_cw_amplitude_dir = patient_dir

# Create the main window
root = tk.Tk()
root.title("Select Patient Data")

# Create a label to prompt the user to select a patient
label = ttk.Label(root, text="Select a patient:")
label.pack(padx=10, pady=10)

# Create a dropdown list with the available patients
patients = ["All Patients"]
patients.extend([f"Patient {i}" for i in range(1, num_patients() + 1)])
patient_dropdown = ttk.Combobox(root, values=patients)
patient_dropdown.current(0)
patient_dropdown.pack(padx=10, pady=10)

# Create a button to submit the selected patient
submit_button = ttk.Button(root, text="Submit", command=choose_patient)
submit_button.pack(padx=10, pady=10)

# Start the main event loop
root.mainloop()

# # Specify the path of the folder you want to check
# folder_path = r'C:\Users\dalto\Downloads\Shlab\Shlab_data'

# # Get the list of all files and folders in the specified folder
# folder_contents = os.listdir(folder_path)

# # Count the number of folders in the specified folder
# folder_count = 0
# for item in folder_contents:
#     item_path = os.path.join(folder_path, item)
#     if os.path.isdir(item_path):
#         folder_count += 1

# # Print the number of folders in the specified folder
# print("Number of folders in", folder_path, "is:", folder_count)