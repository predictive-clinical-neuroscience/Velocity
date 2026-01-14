#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:23:26 2024

@author: johbay
"""
import numpy as np
import os
import pandas as pd
import pickle
import subprocess
import math
import json

def DK_idp_cols():
    mylist = list(['L_bankssts', 
                   'L_caudalanteriorcingulate',
                   'L_caudalmiddlefrontal',
                   'L_cuneus',
                   'L_entorhinal',
                   'L_fusiform',
                   'L_inferiorparietal',
                   'L_inferiortemporal',
                   'L_isthmuscingulate',
                   'L_lateraloccipital',
                   'L_lateralorbitofrontal',
                   'L_lingual',
                   'L_medialorbitofrontal',
                   'L_middletemporal',
                   'L_parahippocampal',
                   'L_paracentral',
                   'L_parsopercularis',
                   'L_parsorbitalis',
                   'L_parstriangularis',
                   'L_pericalcarine',
                   'L_postcentral',
                   'L_posteriorcingulate',
                   'L_precentral',
                   'L_precuneus',
                   'L_rostralanteriorcingulate',
                   'L_rostralmiddlefrontal',
                   'L_superiorfrontal',
                   'L_superiorparietal',
                   'L_superiortemporal',
                   'L_supramarginal',
                   'L_frontalpole',
                   'L_temporalpole',
                   'L_transversetemporal',
                   'L_insula',
                   'R_bankssts',
                   'R_caudalanteriorcingulate',
                   'R_caudalmiddlefrontal',
                   'R_cuneus',
                   'R_entorhinal',
                   'R_fusiform',
                   'R_inferiorparietal',
                   'R_inferiortemporal',
                   'R_isthmuscingulate',
                   'R_lateraloccipital',
                   'R_lateralorbitofrontal',
                   'R_lingual',
                   'R_medialorbitofrontal',
                   'R_middletemporal',
                   'R_parahippocampal',
                   'R_paracentral',
                   'R_parsopercularis',
                   'R_parsorbitalis',
                   'R_parstriangularis',
                   'R_pericalcarine',
                   'R_postcentral',
                   'R_posteriorcingulate',
                   'R_precentral',
                   'R_precuneus',
                   'R_rostralanteriorcingulate',
                   'R_rostralmiddlefrontal',
                   'R_superiorfrontal',
                   'R_superiorparietal',
                   'R_superiortemporal',
                   'R_supramarginal',
                   'R_frontalpole',
                   'R_temporalpole',
                   'R_transversetemporal',
                   'R_insula'])
    return mylist


def idpcols_checker(a_list):
    mylist = DK_idp_cols()
    res = [x for x in a_list + mylist if x not in a_list or x not in mylist]
    return len(res)


# a simple function to quickly load pickle files
def ldpkl(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)



# function to retrieve euler number
def retrieve_eulernumber( subjects):
    
    #Retrieve euler number for OASIS
    #freesurfer_dir = '/project_cephfs/3022017.06/OASIS3/freesurfer/'
    freesurfer_dir = '/project_cephfs/3022017.02/freesurfer/'

    subjects = [temp for temp in os.listdir(freesurfer_dir) 
                if os.path.isdir(os.path.join(freesurfer_dir ,temp))]

    #lines = np.loadtxt('/project_cephfs/3022017.02/freesurfer/folderlist.txt',  delimiter=",", unpack=False)
    #subjects = pd.read_csv('/project_cephfs/3022017.02/freesurfer/folderlist.txt', header=None)


    df = pd.DataFrame(index=subjects, columns=[ 'lh_en','rh_en','avg_en'])

    missing_subjects = []

    for sub in subjects: 
    
        try: 
            bashCommand = 'mris_euler_number '+ freesurfer_dir + sub +'/surf/lh.orig.nofix>' + 'temp_l.txt 2>&1'
            res = subprocess.run(bashCommand, stdout=subprocess.PIPE, shell=True)
            file = open('temp_l.txt', mode = 'r', encoding = 'utf-8-sig')
            lines = file.readlines()
            file.close()
            words = []
            for line in lines:
                line = line.strip()
                words.append([item.strip() for item in line.split(' ')])
            eno_l = np.float32(words[0][12])
            
            bashCommand = 'mris_euler_number '+ freesurfer_dir + sub +'/surf/rh.orig.nofix>' + 'temp_r.txt 2>&1'
            res = subprocess.run(bashCommand, stdout=subprocess.PIPE, shell=True)
            file = open('temp_r.txt', mode = 'r', encoding = 'utf-8-sig')
            lines = file.readlines()
            file.close()
            words = [] 
            for line in lines:
                line = line.strip()
                words.append([item.strip() for item in line.split(' ')])
            eno_r = np.float32(words[0][12])
                            
            df.at[sub, 'lh_en'] = eno_l
            df.at[sub, 'rh_en'] = eno_r
            df.at[sub, 'avg_en'] = (eno_r + eno_l) / 2
                        # print('%d: Subject %s is successfully processed. EN = %f'
                    #        %(s, sub, df.at[sub, 'avg_en']))
        except:
            missing_subjects.append(sub)
                    # print('%d: QC is failed for subject %s.' %(s, sub))
    
    df.to_csv(os.path.join(freesurfer_dir, "euler_number.csv"))
    


def DES_idp_cols():
    mylist= list([
        'lh_G_and_S_frontomargin',
        'lh_G_and_S_occipital_inf',
        'lh_G_and_S_paracentral',
        'lh_G_and_S_subcentral',
        'lh_G_and_S_transv_frontopol',
        'lh_G_and_S_cingul-Ant',
        'lh_G_and_S_cingul-Mid-Ant',
        'lh_G_and_S_cingul-Mid-Post',
        'lh_G_cingul-Post-dorsal',
        'lh_G_cingul-Post-ventral',
        'lh_G_cuneus',
        'lh_G_front_inf-Opercular',
        'lh_G_front_inf-Orbital',
        'lh_G_front_inf-Triangul',
        'lh_G_front_middle',
        'lh_G_front_sup',
        'lh_G_Ins_lg_and_S_cent_ins',
        'lh_G_insular_short',
        'lh_G_occipital_middle',
        'lh_G_occipital_sup',
        'lh_G_oc-temp_lat-fusifor',
        'lh_G_oc-temp_med-Lingual',
        'lh_G_oc-temp_med-Parahip',
        'lh_G_orbital',
        'lh_G_pariet_inf-Angular',
        'lh_G_pariet_inf-Supramar',
        'lh_G_parietal_sup',
        'lh_G_postcentral',
        'lh_G_precentral',
        'lh_G_precuneus',
        'lh_G_rectus',
        'lh_G_subcallosal',
        'lh_G_temp_sup-G_T_transv',
        'lh_G_temp_sup-Lateral',
        'lh_G_temp_sup-Plan_polar',
        'lh_G_temp_sup-Plan_tempo',
        'lh_G_temporal_inf',
        'lh_G_temporal_middle',
        'lh_Lat_Fis-ant-Horizont',
        'lh_Lat_Fis-ant-Vertical',
        'lh_Lat_Fis-post',
        'lh_Pole_occipital',
        'lh_Pole_temporal',
        'lh_S_calcarine',
        'lh_S_central',
        'lh_S_cingul-Marginalis',
        'lh_S_circular_insula_ant',
        'lh_S_circular_insula_inf',
        'lh_S_circular_insula_sup',
        'lh_S_collat_transv_ant',
        'lh_S_collat_transv_post',
        'lh_S_front_inf',
        'lh_S_front_middle',
        'lh_S_front_sup',
        'lh_S_interm_prim-Jensen',
        'lh_S_intrapariet_and_P_trans',
        'lh_S_oc_middle_and_Lunatus',
        'lh_S_oc_sup_and_transversal',
        'lh_S_occipital_ant',
        'lh_S_oc-temp_lat',
        'lh_S_oc-temp_med_and_Lingual',
        'lh_S_orbital_lateral',
        'lh_S_orbital_med-olfact',
        'lh_S_orbital-H_Shaped',
        'lh_S_parieto_occipital',
        'lh_S_pericallosal',
        'lh_S_postcentral',
        'lh_S_precentral-inf-part',
        'lh_S_precentral-sup-part',
        'lh_S_suborbital',
        'lh_S_subparietal',
        'lh_S_temporal_inf',
        'lh_S_temporal_sup',
        'lh_S_temporal_transverse',
        'rh_G_and_S_frontomargin',
        'rh_G_and_S_occipital_inf',
        'rh_G_and_S_paracentral',
        'rh_G_and_S_subcentral',
        'rh_G_and_S_transv_frontopol',
        'rh_G_and_S_cingul-Ant',
        'rh_G_and_S_cingul-Mid-Ant',
        'rh_G_and_S_cingul-Mid-Post',
        'rh_G_cingul-Post-dorsal',
        'rh_G_cingul-Post-ventral',
        'rh_G_cuneus',
        'rh_G_front_inf-Opercular',
        'rh_G_front_inf-Orbital',
        'rh_G_front_inf-Triangul',
        'rh_G_front_middle',
        'rh_G_front_sup',
        'rh_G_Ins_lg_and_S_cent_ins',
        'rh_G_insular_short',
        'rh_G_occipital_middle',
        'rh_G_occipital_sup',
        'rh_G_oc-temp_lat-fusifor',
        'rh_G_oc-temp_med-Lingual',
        'rh_G_oc-temp_med-Parahip',
        'rh_G_orbital',
        'rh_G_pariet_inf-Angular',
        'rh_G_pariet_inf-Supramar',
        'rh_G_parietal_sup',
        'rh_G_postcentral',
        'rh_G_precentral',
        'rh_G_precuneus',
        'rh_G_rectus',
        'rh_G_subcallosal',
        'rh_G_temp_sup-G_T_transv',
        'rh_G_temp_sup-Lateral',
        'rh_G_temp_sup-Plan_polar',
        'rh_G_temp_sup-Plan_tempo',
        'rh_G_temporal_inf',
        'rh_G_temporal_middle',
        'rh_Lat_Fis-ant-Horizont',
        'rh_Lat_Fis-ant-Vertical',
        'rh_Lat_Fis-post',
        'rh_Pole_occipital',
        'rh_Pole_temporal',
        'rh_S_calcarine',
        'rh_S_central',
        'rh_S_cingul-Marginalis',
        'rh_S_circular_insula_ant',
        'rh_S_circular_insula_inf',
        'rh_S_circular_insula_sup',
        'rh_S_collat_transv_ant',
        'rh_S_collat_transv_post',
        'rh_S_front_inf',
        'rh_S_front_middle',
        'rh_S_front_sup',
        'rh_S_interm_prim-Jensen',
        'rh_S_intrapariet_and_P_trans',
        'rh_S_oc_middle_and_Lunatus',
        'rh_S_oc_sup_and_transversal',
        'rh_S_occipital_ant',
        'rh_S_oc-temp_lat',
        'rh_S_oc-temp_med_and_Lingual',
        'rh_S_orbital_lateral',
        'rh_S_orbital_med-olfact',
        'rh_S_orbital-H_Shaped',
        'rh_S_parieto_occipital',
        'rh_S_pericallosal',
        'rh_S_postcentral',
        'rh_S_precentral-inf-part',
        'rh_S_precentral-sup-part',
        'rh_S_suborbital',
        'rh_S_subparietal',
        'rh_S_temporal_inf',
        'rh_S_temporal_sup',
        'rh_S_temporal_transverse']
             )
    return mylist




def save_to_json(filename, **kwargs):
    try:
        # Load existing data from the file if it exists
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        # Update the data with new key-value pairs
        data.update(kwargs)

        # Save the updated data back to the file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Data successfully saved to {filename}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def SC_idp_cols_new():
     mylist= list([
         'Left-Lateral-Ventricle',
         'Left-Inf-Lat-Vent',
         'Left-Cerebellum-White-Matter',
         'Left-Cerebellum-Cortex',
         'Left-Thalamus',
         'Left-Caudate',
         'Left-Putamen',
         'Left-Pallidum',
         '3rd-Ventricle',
         '4th-Ventricle',
         'Brain-Stem',
         'Left-Hippocampus',
         'Left-Amygdala',
         'CSF',
         'Left-Accumbens-area',
         'Left-VentralDC',
         'Left-vessel',
         'Left-choroid-plexus',
         'Right-Lateral-Ventricle',
         'Right-Inf-Lat-Vent',
         'Right-Cerebellum-White-Matter',
         'Right-Cerebellum-Cortex',
         'Right-Thalamus',
         'Right-Caudate',
         'Right-Putamen',
         'Right-Pallidum',
         'Right-Hippocampus',
         'Right-Amygdala',
         'Right-Accumbens-area',
         'Right-VentralDC',
         'Right-vessel',
         'Right-choroid-plexus',
         '5th-Ventricle',
         'WM-hypointensities',
         #'Left-WM-hypointensities',
         #'Right-WM-hypointensities',
         'non-WM-hypointensities',
         #'Left-non-WM-hypointensities',
         #'Right-non-WM-hypointensities',
         'Optic-Chiasm',
         'CC_Posterior',
         'CC_Mid_Posterior',
         'CC_Central',
         'CC_Mid_Anterior',
         'CC_Anterior',
         'BrainSegVol',
         #'BrainSegVolNotVent',
         'lhCortexVol',
         'rhCortexVol',
         #'CortexVol',
         #'lhCerebralWhiteMatterVol',
         #'rhCerebralWhiteMatterVol',
         #'CerebralWhiteMatterVol',
         'SubCortGrayVol',
         'TotalGrayVol',
         'SupraTentorialVol',
         'SupraTentorialVolNotVent',
         #'MaskVol',
         'BrainSegVol-to-eTIV',
         'MaskVol-to-eTIV',
         #'lhSurfaceHoles',
         #'rhSurfaceHoles',
         #'SurfaceHoles',
         'EstimatedTotalIntraCranialVol'])
     return(mylist)
             
        
def SC_idp_cols():
    mylist= list([
        'Left-Lateral-Ventricle',
        'Left-Inf-Lat-Vent',
        'Left-Cerebellum-White-Matter',
        'Left-Cerebellum-Cortex',
        'Left-Thalamus',
        'Left-Caudate',
        'Left-Putamen',
        'Left-Pallidum',
        '3rd-Ventricle',
        '4th-Ventricle',
        'Brain-Stem',
        'Left-Hippocampus',
        'Left-Amygdala',
        'CSF',
        'Left-Accumbens-area',
        'Left-VentralDC',
        'Left-vessel',
        'Left-choroid-plexus',
        'Right-Lateral-Ventricle',
        'Right-Inf-Lat-Vent',
        'Right-Cerebellum-White-Matter',
        'Right-Cerebellum-Cortex',
        'Right-Thalamus',
        'Right-Caudate',
        'Right-Putamen',
        'Right-Pallidum',
        'Right-Hippocampus',
        'Right-Amygdala',
        'Right-Accumbens-area',
        'Right-VentralDC',
        'Right-vessel',
        'Right-choroid-plexus',
        '5th-Ventricle',
        'WM-hypointensities',
        #'Left-WM-hypointensities',
        #'Right-WM-hypointensities',
        'non-WM-hypointensities',
        #'Left-non-WM-hypointensities',
        #'Right-non-WM-hypointensities',
        'Optic-Chiasm',
        'CC_Posterior',
        'CC_Mid_Posterior',
        'CC_Central',
        'CC_Mid_Anterior',
        'CC_Anterior',
        'BrainSegVol',
        #'BrainSegVolNotVent',
        'lhCortexVol',
        'rhCortexVol',
        #'CortexVol',
        #'lhCerebralWhiteMatterVol',
        #'rhCerebralWhiteMatterVol',
        #'CerebralWhiteMatterVol',
        'SubCortGrayVol',
        'TotalGrayVol',
        'SupraTentorialVol',
        'SupraTentorialVolNotVent',
        #'MaskVol',
        'BrainSegVol-to-eTIV',
        'MaskVol-to-eTIV',
        'lhSurfaceHoles',
        'rhSurfaceHoles',
        #'SurfaceHoles',
        'EstimatedTotalIntraCranialVol'])
    return(mylist)
     

def reduced_sc_idp_cols():
    mylist=list([
        '3rd-Ventricle',
        '4th-Ventricle',
        'Brain-Stem',
        'CSF',
        'EstimatedTotalIntraCranialVol',
        'Left-Accumbens-area',
        'Left-Amygdala',
        'Left-Caudate',
        'Left-Cerebellum-Cortex',
        'Left-Cerebellum-White-Matter',
        'Left-Hippocampus',
        'Left-Inf-Lat-Vent',
        'Left-Lateral-Ventricle',
        'Left-Pallidum',
        'Left-Putamen',
        #'Left-Thalamus-Proper',
        'Left-VentralDC',
        'Left-choroid-plexus',
        'Left-vessel',
        'Right-Accumbens-area',
        'Right-Amygdala',
        'Right-Caudate',
        'Right-Cerebellum-Cortex',
        'Right-Cerebellum-White-Matter',
        'Right-Hippocampus',
        'Right-Inf-Lat-Vent',
        'Right-Lateral-Ventricle',
        'Right-Pallidum',
        'Right-Putamen',
        #'Right-Thalamus-Proper',
        'Right-VentralDC',
        'Right-choroid-plexus',
        'Right-vessel',
        'SubCortGrayVol',
        'SupraTentorialVol',
        'SupraTentorialVolNotVent',
        'TotalGrayVol'])
    return mylist

#%%
def remove_outliers_and_adjust(X_train, Y_train, Z_train, threshold=7):
# List to store the indices of the rows to keep (not outliers)
   valid_indices = []

# Loop over each column in Y_train

   mean = Y_train.mean()
   std = Y_train.std()

# Identify the rows where the values are more than `threshold` standard deviations away from the mean
   non_outliers = np.abs(Y_train - mean) <= threshold * std

   valid_indices = non_outliers

# Filter out the rows in Y_train, X_train, and Z_train using the valid indices
   Y_train_cleaned = Y_train[valid_indices]
   X_train_cleaned = X_train[valid_indices]
   Z_train_cleaned = Z_train[valid_indices]

# Convert the cleaned DataFrames to NumPy arrays
   X_train_array = X_train_cleaned.to_numpy()
   Y_train_array = Y_train_cleaned.to_numpy()
   Z_train_array = Z_train_cleaned.to_numpy()

# Return the resulting NumPy arrays
   return X_train_array, Y_train_array, Z_train_array

#%%
def count_site_occurrences(train, test=None):
    """
    For each unique site_id in the train DataFrame, print the count of occurrences in the train and optionally in the test DataFrame.
    
    Parameters:
    - train (pd.DataFrame): The training dataset containing the site_id column.
    - test (pd.DataFrame, optional): The testing dataset containing the site_id column. If not provided, only train data will be used.
    """
    # Get the unique site_ids from the train dataset
    sites = train.site_id.unique()
    
    # Iterate over each unique site_id
    for i, s in enumerate(sites):
        # Find the indices where site_id matches the current site in train
        idx_train = train['site_id'] == s
        
        # Print the train count
        print(f"Site {s}: Train count = {sum(idx_train)}", end="")

        # If test DataFrame is provided, count occurrences in test as well
        if test is not None:
            idx_test = test['site_id'] == s
            print(f", Test count = {sum(idx_test)}")
        else:
            print()  # Just move to the next line if no test DataFrame is provided

def drop_rows_with_nans_in_columns(df: pd.DataFrame, columns: list = ["site_id", "age", "sex"]) -> pd.DataFrame:
    """
    Remove all rows from the DataFrame that contain NaNs in any of the specified columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of columns to check for NaNs.

    Returns:
    - pd.DataFrame: A new DataFrame with the rows removed.
    """
    return df.dropna(subset=columns)

def site_dictionary():
   site_to_id = {
    800: ("OASIS2", "OASIS2"),
    401: ("OASIS3_401", "OASIS3"),
    402: ("OASIS3_402", "OASIS3"),
    403: ("OASIS3_403", "OASIS3"),
    404: ("OASIS3_404", "OASIS3"),
    405: ("OASIS3_405", "OASIS3"),
    700: ("NOTTINGHAM", "IMAGEN"),
    701: ("PARIS", "IMAGEN"),
    702: ("DRESDEN", "IMAGEN"),
    703: ("PARIS", "IMAGEN"),
    704: ("MANNHEIM", "IMAGEN"),
    705: ("DUBLIN", "IMAGEN"),
    706: ("LONDON", "IMAGEN"),
    707: ("HAMBURG", "IMAGEN"),
    54001: ("abcd_01", "ABCD"),
    54002: ("abcd_02", "ABCD"),
    54003: ("abcd_03", "ABCD"),
    54004: ("abcd_04", "ABCD"),
    54005: ("abcd_05", "ABCD"),
    54006: ("abcd_06", "ABCD"),
    54007: ("abcd_07", "ABCD"),
    54008: ("abcd_08", "ABCD"),
    54009: ("abcd_09", "ABCD"),
    54010: ("abcd_10", "ABCD"),
    54011: ("abcd_11", "ABCD"),
    54012: ("abcd_12", "ABCD"),
    54013: ("abcd_13", "ABCD"),
    54014: ("abcd_14", "ABCD"),
    54015: ("abcd_15", "ABCD"),
    54016: ("abcd_16", "ABCD"),
    54017: ("abcd_17", "ABCD"),
    54018: ("abcd_18", "ABCD"),
    54019: ("abcd_19", "ABCD"),
    54020: ("abcd_20", "ABCD"),
    54021: ("abcd_21", "ABCD"),
    54022: ("abcd_22", "ABCD"),
    11025: ("UKB_11025", "UKB"),
    11026: ("UKB_11026", "UKB"),
    11027: ("UKB_11027", "UKB"),
    501: ("UMich_SAD", "life_tr"),
    502: ("UMich_CWS", "life_tr"),
    503: ("top", "life_tr"),
    504: ("SWA", "life_tr"),
    505: ("ABIDE_USM", "life_tr"),
    506: ("ABIDE_KKI", "life_tr"),
    507: ("ABIDE_NYU", "life_tr"),
    508: ("HUH", "life_tr"),
    509: ("HRC", "life_tr"),
    510: ("HKH", "life_tr"),
    511: ("COI", "life_tr"),
    512: ("UMich_MLS", "life_tr"),
    513: ("KUT", "life_tr"),
    514: ("delta", "life_tr"),
    515: ("KTT", "life_tr"),
    516: ("HCP_EP_IU", "life_tr"),
    517: ("UMich_IMPs", "life_tr"),
    518: ("CNP-35343.0", "life_tr"),
    519: ("hcp_ya", "life_tr"),
    520: ("UMich_MTwins", "life_tr"),
    521: ("CNP-35426.0", "life_tr"),
    522: ("ADD200_NYU", "life_tr"),
    523: ("UTO", "life_tr"),
    524: ("ADD200_KKI", "life_tr"),
    525: ("ATV", "life_tr"),
    526: ("CIN", "life_tr"),
    527: ("HCP_EP_BWH", "life_tr"),
    528: ("SWU_SLIM_ses1", "life_tr"),
    529: ("ABIDE_GU", "life_tr"),
    530: ("HCP_EP_MGH", "life_tr"),
    531: ("HCP_EP_McL", "life_tr"),
    532: ("KCL", "life_tr"),
    533: ("pnc", "life_tr"),
    534: ("nki", "life_tr"),
    535: ("AOMIC_1000", "life_tr"),
    536: ("AOMIC_PIPO2", "life_tr"),
    537: ("cam", "life_tr"),
    538: ("HCP_A_UM", "life_tr"),
    539: ("HCP_A_WU", "life_tr"),
    540: ("HCP_A_MGH", "life_tr"),
    541: ("HCP_A_UCLA", "life_tr"),
    542: ("HCP_D_WU", "life_tr"),
    543: ("HCP_D_UCLA", "life_tr"),
    544: ("HCP_D_UM", "life_tr"),
    545: ("HCP_D_MGH", "life_tr"),
    546: ("ixi", "life_tr"),
    547: ("CMI_RU", "life_tr"),
    548: ("CMI_SI", "life_tr"),
    549: ("CMI_CBIC", "life_tr"),
    550: ("UMich_SZG", "life_tr"),
    551: ("ds001734", "life_tr"),
    552: ("ds002236", "life_tr"),
    553: ("ds002330", "life_tr"),
    554: ("ds002345", "life_tr"),
    555: ("ds002731", "life_tr"),
    556: ("ds002837", "life_tr"),
    557: ("UCDavis", "life_tr"),
    558: ("ping_c", "life_tr"),
    559: ("ping_i", "life_tr"),
    560: ("ping_a", "life_tr"),
    561: ("ping_d", "life_tr"),
    562: ("ping_h", "life_tr"),
    563: ("ping_f", "life_tr"),
    564: ("ping_j", "life_tr"),
    932: ("ADNI_1","ADNI"),
    922: ("ADNI_2","ADNI"),
    973: ("ADNI_3","ADNI"),
    923: ("ADNI_4","ADNI"),
    913: ("ADNI_5","ADNI"),
    909: ("ADNI_6","ADNI"),
    1037: ("ADNI_7","ADNI"),
    1016: ("ADNI_8","ADNI"),
    1841: ("ADNI_9","ADNI"),
    999: ("ADNI_10","ADNI"),
    1028: ("ADNI_11","ADNI"),
    918: ("ADNI_12","ADNI"),
    1014: ("ADNI_13","ADNI"),
    933: ("ADNI_14","ADNI"),
    902: ("ADNI_15","ADNI"),
    1041: ("ADNI_16","ADNI"),
    941: ("ADNI_17","ADNI"),
    1033: ("ADNI_18","ADNI"),
    957: ("ADNI_19","ADNI"),
    911: ("ADNI_20","ADNI"),
    906: ("ADNI_21","ADNI"),
    937: ("ADNI_22","ADNI"),
    920: ("ADNI_23","ADNI"),
    962: ("ADNI_24","ADNI"),
    919: ("ADNI_25","ADNI"),
    936: ("ADNI_26","ADNI"),
    912: ("ADNI_27","ADNI"),
    1053: ("ADNI_28","ADNI"),
    914: ("ADNI_29","ADNI"),
    924: ("ADNI_30","ADNI"),
    982: ("ADNI_31","ADNI"),
    910: ("ADNI_32","ADNI"),
    1035: ("ADNI_33","ADNI"),
    
    }
   return site_to_id