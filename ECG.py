
# 10/24/2020

import os, os.path
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import torch.utils.data
import pandas as pd
import pathlib
import math
verbose = False


def rolling_average (array,value_length):
    new_array = np.zeros((1,5000,12))
    assert array.shape == (1,5000,12), "array is not shape (1,2500,12)"
    for i in range(0,12):
        new_array[0,:,i]=pd.Series(array[0][:,i]).rolling(window=value_length,min_periods=1).mean() #min_periods ensure no NaNs before value_length fulfilled
    return new_array


# +
# 2, 4 , 8, 16, 
# -

5000//64


def plot(array,color = 'blue'):
    lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    plt.rcParams["figure.figsize"] = [16,9]
    
    fig, axs = plt.subplots(len(lead_order))
    fig.suptitle("array")
    # rolling_arr = rolling_average(array,15)
    if array.shape == (5000, 12):
        for i in range(0,12):
            axs[i].plot(array[:2500,i],label = 'window')
            # axs[i].plot(array[::2,i],label = 'downsample')
            # axs[i].plot(rolling_arr[:2500,i],label = 'rolling')
            axs[i].set(ylabel=str(lead_order[i]))
    elif array.shape == (12, 5000):
        for i in range(0,12):
            axs[i].plot(array[i,:2500],label = 'window')
            # axs[i].plot(array[i,::2],label = 'downsample')
            # axs[i].plot(rolling_arr[i,:],label = 'rolling')
            axs[i].set(ylabel=str(lead_order[i]))
    elif array.shape == (1,5000,12):
        for i in range(0,12):
            axs[i].plot(array[0,:5000,i],label = 'window')
            # axs[i].plot(array[0,::2,i],label = 'downsample')
            # axs[i].plot(rolling_arr[0,:2500,i],label = 'rolling')
            axs[i].set(ylabel=str(lead_order[i]))
    elif array.shape == (1,1,5000,12):
        for i in range(0,12):
            axs[i].plot(array[0,0,:5000,i],label = 'window')
            # axs[i].plot(array[0,0,::2,i],label = 'downsample')
            # axs[i].plot(rolling_arr[0,0,:2500,i],label = 'rolling')
            axs[i].set(ylabel=str(lead_order[i]))
    else:
        print("ECG shape not valid: ",array.shape)
    
    plt.show()


class ECG_loader(torch.utils.data.Dataset):
    
    def __init__(self, root=None,csv = None,bootstrap = False,sliding = False,downsample=0,rolling = 0,plot_data = False,additional_inputs = None,target = 'Mortality'):
        if root is None:
            root = "EchoNet_ECG_waveforms"
        if csv is None:
            csv =  "EFFileList.csv"
        self.folder = pathlib.Path(root)
        self.file_list = pd.read_csv(csv)
        if bootstrap:
            self.file_list = self.file_list.sample(frac=0.5, replace=True).reset_index(drop=True)
        self.sliding = sliding
        self.downsample = downsample
        self.rolling = rolling
        self.plot_data = plot_data
        self.additional_inputs = additional_inputs
        self.target = target
    def __getitem__(self, index):
        if 'filename' in self.file_list.columns:
            fname = self.file_list.filename[index%len(self.file_list.index)]
        if 'Filename' in self.file_list.columns:
            fname = self.file_list.Filename[index%len(self.file_list.index)]
        waveform = np.load(os.path.join(self.folder, fname))
        if waveform.shape == (5000,12):
            waveform = np.expand_dims(waveform,axis=0)
        x = []
        if not self.additional_inputs is None:
            for i in self.additional_inputs:
                x.append(self.file_list[i][index%len(self.file_list.index)])
        start = np.random.randint(2499)
        if self.rolling != 0:
            waveform = rolling_average(waveform,self.rolling)
        if self.plot_data == True:
            plot(waveform,color = 'orange')
            plt.show()
        if self.sliding:
            waveform = waveform[:,start:start+2500]
        if self.downsample>0:
            waveform = waveform[:,::self.downsample,:]
            
        target = self.file_list[self.target][index]
        target = torch.FloatTensor([target])
        waveform = np.transpose(waveform[0,:,:],(1,0))
        waveform = torch.FloatTensor(waveform)
        
        if not self.additional_inputs is None:
            return (waveform,torch.FloatTensor(x)), target
        else:
            return waveform, target
    

    def __len__(self):

        return math.ceil(len(self.file_list.index))


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
