#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:27:03 2023

@author: johbay
"""
import os
import numpy as np
import pickle 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

#%% Get covariates



#peds_demographics01.txt 

#%%
def print_columns(data_table):
    for col in data_table:
        print (col)
        
#%%    
def prepare_model_inputs_train_test(processing_dir, df, test_ratio = 0.5):
    
    if not os.path.isdir(processing_dir):
        os.mkdir(processing_dir)
        log_dir = processing_dir + 'log/'
        os.mkdir(log_dir)
        os.mkdir(processing_dir + 'Models/')
    
    age = df['age'].to_numpy(dtype=float)
    labels = df['diagnosis'].to_numpy(dtype=int)
    
    cortical_thickness = df.iloc[:,7:211]
    cortical_thickness = cortical_thickness.apply(lambda x: x.fillna(x.mean()),axis='index').to_numpy(dtype=float)
    columnsTitles=['site_id','sex']
    batch_effects=df.reindex(columns=columnsTitles)
    batch_effects = batch_effects.to_numpy(dtype=int)
    u = np.unique(batch_effects[:,0])
    for i in range(len(u)):
       batch_effects[batch_effects[:,0]==u[i],0]=i    
    
    # This just does a random split (i.e. not stratified)
    te = np.random.uniform(size=len(df)) < test_ratio
    tr = ~te    
    training_idx = np.where(np.bitwise_and(labels==0, tr))[0] 
    testing_idx = np.where(np.bitwise_and(labels==0, te))[0] 
    
    # df['test'] = False
    # te = df['test'].to_numpy()
    # for s in df['site_id'].unique():
    #     sid_0 = np.where((df['site_id'] == s) & (df['gender'] == 0))[0]
    #     sid_1 = np.where((df['site_id'] == s) & (df['gender'] == 1))[0]
        
    #     te_sid_0 = sid_0[:math.ceil(len(sid_0) * test_ratio)]
    #     te_sid_1 = sid_1[:math.ceil(len(sid_1) * test_ratio)]
    #     print(s, len(te_sid_0), '/', len(sid_0), ',',  len(te_sid_1),'/', len(sid_1))
    #     te[te_sid_0] = True
    #     te[te_sid_1] = True
    
    tr = ~te    
    training_idx = np.where(np.bitwise_and(labels==0, tr))[0] 
    testing_idx = np.where(np.bitwise_and(labels==0, te))[0] 
    
    X_train = age[training_idx]/100
    Y_train = cortical_thickness[training_idx,:]
    batch_effects_train = batch_effects[training_idx]
    with open(processing_dir + 'X_train.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_train), file)
    with open(processing_dir + 'Y_train.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(Y_train), file) 
    with open(processing_dir + 'trbefile.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_train), file) 
    df.iloc[training_idx].to_csv(os.path.join(processing_dir,'df_tr.csv'))
    
    X_test = age[testing_idx]/100
    Y_test = cortical_thickness[testing_idx,:]
    batch_effects_test = batch_effects[testing_idx]
    with open(processing_dir + 'X_test.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_test), file)
    with open(processing_dir + 'Y_test.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(Y_test), file) 
    with open(processing_dir + 'tsbefile.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_test), file) 
    df.iloc[testing_idx].to_csv(os.path.join(processing_dir,'df_te.csv'))
  
        
#%%
def adapt_lifespan(lifespan):
    lifespan['group']= 0
    lifespan = lifespan.rename(columns={'group':'diagnosis'})
    #re-categorize site
    le = LabelEncoder()
    lifespan['site_id'] = le.fit_transform(lifespan['site'])
    return(lifespan)
        
#%%
# a simple function to quickly load pickle files
def ldpkl(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)
#%%
def features_dic(which_dataset):
    match  which_dataset:
        case "replication":
            features = ['lh_G&S_frontomargin_thickness',
                'lh_G&S_occipital_inf_thickness',
                'lh_G&S_paracentral_thickness',
                'lh_G&S_subcentral_thickness',
                'lh_G&S_transv_frontopol_thickness',
                'lh_G&S_cingul-Ant_thickness',
                'lh_G&S_cingul-Mid-Ant_thickness',
                'lh_G&S_cingul-Mid-Post_thickness',
                'lh_G_cingul-Post-dorsal_thickness',
                'lh_G_cingul-Post-ventral_thickness',
                'lh_G_cuneus_thickness',
                'lh_G_front_inf-Opercular_thickness',
                'lh_G_front_inf-Orbital_thickness',
                'lh_G_front_inf-Triangul_thickness',
                'lh_G_front_middle_thickness',
                'lh_G_front_sup_thickness',
                'lh_G_Ins_lg&S_cent_ins_thickness',
                'lh_G_insular_short_thickness',
                'lh_G_occipital_middle_thickness',
                'lh_G_occipital_sup_thickness',
                'lh_G_oc-temp_lat-fusifor_thickness',
                'lh_G_oc-temp_med-Lingual_thickness',
                'lh_G_oc-temp_med-Parahip_thickness',
                'lh_G_orbital_thickness',
                'lh_G_pariet_inf-Angular_thickness',
                'lh_G_pariet_inf-Supramar_thickness',
                'lh_G_parietal_sup_thickness',
                'lh_G_postcentral_thickness',
                'lh_G_precentral_thickness',
                'lh_G_precuneus_thickness',
                'lh_G_rectus_thickness',
                'lh_G_subcallosal_thickness',
                'lh_G_temp_sup-G_T_transv_thickness',
                'lh_G_temp_sup-Lateral_thickness',
                'lh_G_temp_sup-Plan_polar_thickness',
                'lh_G_temp_sup-Plan_tempo_thickness',
                'lh_G_temporal_inf_thickness',
                'lh_G_temporal_middle_thickness',
                'lh_Lat_Fis-ant-Horizont_thickness',
                'lh_Lat_Fis-ant-Vertical_thickness',
                'lh_Lat_Fis-post_thickness',
                'lh_Pole_occipital_thickness',
                'lh_Pole_temporal_thickness',
                'lh_S_calcarine_thickness',
                'lh_S_central_thickness',
                'lh_S_cingul-Marginalis_thickness',
                'lh_S_circular_insula_ant_thickness',
                'lh_S_circular_insula_inf_thickness',
                'lh_S_circular_insula_sup_thickness',
                'lh_S_collat_transv_ant_thickness',
                'lh_S_collat_transv_post_thickness',
                'lh_S_front_inf_thickness',
                'lh_S_front_middle_thickness',
                'lh_S_front_sup_thickness',
                'lh_S_interm_prim-Jensen_thickness',
                'lh_S_intrapariet&P_trans_thickness',
                'lh_S_oc_middle&Lunatus_thickness',
                'lh_S_oc_sup&transversal_thickness',
                'lh_S_occipital_ant_thickness',
                'lh_S_oc-temp_lat_thickness',
                'lh_S_oc-temp_med&Lingual_thickness',
                'lh_S_orbital_lateral_thickness',
                'lh_S_orbital_med-olfact_thickness',
                'lh_S_orbital-H_Shaped_thickness',
                'lh_S_parieto_occipital_thickness',
                'lh_S_pericallosal_thickness',
                'lh_S_postcentral_thickness',
                'lh_S_precentral-inf-part_thickness',
                'lh_S_precentral-sup-part_thickness',
                'lh_S_suborbital_thickness',
                'lh_S_subparietal_thickness',
                'lh_S_temporal_inf_thickness',
                'lh_S_temporal_sup_thickness',
                'lh_S_temporal_transverse_thickness',
                'lh_MeanThickness_thickness',
                'rh_G&S_frontomargin_thickness',
                'rh_G&S_occipital_inf_thickness',
                'rh_G&S_paracentral_thickness',
                'rh_G&S_subcentral_thickness',
                'rh_G&S_transv_frontopol_thickness',
                'rh_G&S_cingul-Ant_thickness',
                'rh_G&S_cingul-Mid-Ant_thickness',
                'rh_G&S_cingul-Mid-Post_thickness',
                'rh_G_cingul-Post-dorsal_thickness',
                'rh_G_cingul-Post-ventral_thickness',
                'rh_G_cuneus_thickness',
                'rh_G_front_inf-Opercular_thickness',
                'rh_G_front_inf-Orbital_thickness',
                'rh_G_front_inf-Triangul_thickness',
                'rh_G_front_middle_thickness',
                'rh_G_front_sup_thickness',
                'rh_G_Ins_lg&S_cent_ins_thickness',
                'rh_G_insular_short_thickness',
                'rh_G_occipital_middle_thickness',
                'rh_G_occipital_sup_thickness',
                'rh_G_oc-temp_lat-fusifor_thickness',
                'rh_G_oc-temp_med-Lingual_thickness',
                'rh_G_oc-temp_med-Parahip_thickness',
                'rh_G_orbital_thickness',
                'rh_G_pariet_inf-Angular_thickness',
                'rh_G_pariet_inf-Supramar_thickness',
                'rh_G_parietal_sup_thickness',
                'rh_G_postcentral_thickness',
                'rh_G_precentral_thickness',
                'rh_G_precuneus_thickness',
                'rh_G_rectus_thickness',
                'rh_G_subcallosal_thickness',
                'rh_G_temp_sup-G_T_transv_thickness',
                'rh_G_temp_sup-Lateral_thickness',
                'rh_G_temp_sup-Plan_polar_thickness',
                'rh_G_temp_sup-Plan_tempo_thickness',
                'rh_G_temporal_inf_thickness',
                'rh_G_temporal_middle_thickness',
                'rh_Lat_Fis-ant-Horizont_thickness',
                'rh_Lat_Fis-ant-Vertical_thickness',
                'rh_Lat_Fis-post_thickness',
                'rh_Pole_occipital_thickness',
                'rh_Pole_temporal_thickness',
                'rh_S_calcarine_thickness',
                'rh_S_central_thickness',
                'rh_S_cingul-Marginalis_thickness',
                'rh_S_circular_insula_ant_thickness',
                'rh_S_circular_insula_inf_thickness',
                'rh_S_circular_insula_sup_thickness',
                'rh_S_collat_transv_ant_thickness',
                'rh_S_collat_transv_post_thickness',
                'rh_S_front_inf_thickness',
                'rh_S_front_middle_thickness',
                'rh_S_front_sup_thickness',
                'rh_S_interm_prim-Jensen_thickness',
                'rh_S_intrapariet&P_trans_thickness',
                'rh_S_oc_middle&Lunatus_thickness',
                'rh_S_oc_sup&transversal_thickness',
                'rh_S_occipital_ant_thickness',
                'rh_S_oc-temp_lat_thickness',
                'rh_S_oc-temp_med&Lingual_thickness',
                'rh_S_orbital_lateral_thickness',
                'rh_S_orbital_med-olfact_thickness',
                'rh_S_orbital-H_Shaped_thickness',
                'rh_S_parieto_occipital_thickness',
                'rh_S_pericallosal_thickness',
                'rh_S_postcentral_thickness',
                'rh_S_precentral-inf-part_thickness',
                'rh_S_precentral-sup-part_thickness',
                'rh_S_suborbital_thickness',
                'rh_S_subparietal_thickness',
                'rh_S_temporal_inf_thickness',
                'rh_S_temporal_sup_thickness',
                'rh_S_temporal_transverse_thickness',
                'rh_MeanThickness_thickness',
                'Left-Lateral-Ventricle',
                'Left-Inf-Lat-Vent',
                'Left-Cerebellum-White-Matter',
                'Left-Cerebellum-Cortex',
                'Left-Thalamus-Proper',
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
                'Right-Thalamus-Proper',
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
                'non-WM-hypointensities',
                'Optic-Chiasm',
                'CC_Posterior',
                'CC_Mid_Posterior',
                'CC_Central',
                'CC_Mid_Anterior',
                'CC_Anterior',
                'BrainSegVol',
                'lhCortexVol',
                'rhCortexVol',
                'SubCortGrayVol',
                'TotalGrayVol',
                'SupraTentorialVol',
                'SupraTentorialVolNotVent',
                'BrainSegVol-to-eTIV',
                'MaskVol-to-eTIV',
                'lhSurfaceHoles',
                'rhSurfaceHoles',
                'EstimatedTotalIntraCranialVol',
                'avg_thickness'
    ]           
        case "more_chains":
            features = ['Right-Cerebellum-White-Matter',
                            'EstimatedTotalIntraCranialVol',
                            'Right-Lateral-Ventricle',
                            'WM-hypointensities',
                            'rh_S_interm_prim-Jensen_thickness', 
                            'Brain-Stem','log_WM-hypointensities']
    return features
#%%
def grab_colnames(data, begin:int, end:int):
    colnames = data.iloc[:, begin:end]
    return colnames
