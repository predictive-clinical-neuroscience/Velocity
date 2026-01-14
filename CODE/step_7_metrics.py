#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:10:09 2025

@author: johbay
This script loads  the quaility metrix from the new model 
"""
import pandas as pd

def new_header(dataframe):
    """
    Parameters
    ----------
    dataframe : pandas data frame
        takes pandas data frame and makes the first row a colum header.

    Returns
    -------
    pandas data frame, rounded.
    """
    new_header = dataframe.iloc[0] 
    dataframe = dataframe[1:] #take the data less the header row
    dataframe.columns = new_header #set the header row as the df header
    return dataframe
    
def make_numeric_and_round(dataframe):
    """
    Parameters
    ----------
    dataframe : pandas data frame
        DESCRIPTION.

    Returns
    -------
    pandas data frame, rounded.
    """
    dataframe = dataframe.apply(pd.to_numeric)
    dataframe = dataframe.round(decimals=3)
    return dataframe

DK_test = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_DK_all_regions/results/statistics_test.csv")
SC_test = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_SC_all_regions/results/statistics_test.csv")
DES_test = pd.read_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_DES_all_regions/results/statistics_test.csv")


DK_test_long = DK_test.transpose()
SC_test_long = SC_test.transpose()

SC_test_long = new_header(SC_test_long)
SC_test_long = make_numeric_and_round(SC_test_long)

DK_test_long = new_header(DK_test_long)
DK_test_long = make_numeric_and_round(DK_test_long)

DK_test_long.to_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_DK_all_regions/results/statistics_test_transpose.csv")
SC_test_long.to_csv("/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_SC_all_regions/results/statistics_test_transpose.csv")





