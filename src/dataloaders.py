import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
from .utils import list_s3_files, get_cube_from_bytestream, zscore, generate_3d_gaussian_psf

class SyntheticSeismicWithGroundTruth(Dataset):
    """Dataset class for loading 3D seismic cubes with ground truth fault masks"""
    def __init__(self, bucket_name, seismic_path, fault_path):
        super().__init__()
        self.bucket_name = bucket_name
        self.list_seismic_files = list_s3_files(bucket_name, seismic_path, '.dat')
        self.list_fault_files = list_s3_files(bucket_name, fault_path, '.dat')
    
    def __len__(self):
        return len(self.list_seismic_files)
    
    def __getitem__(self, index):
        # extract the seismic and fault cubes
        seismic = get_cube_from_bytestream(self.bucket_name, self.list_seismic_files[index])
        fault = get_cube_from_bytestream(self.bucket_name, self.list_fault_files[index])
        
        # transpose dimensions so depth appears as first axis
        seismic = seismic.transpose(2,0,1)
        fault = fault.transpose(2,0,1)
        
        # add multiplicative gaussian noise
        noise = generate_3d_gaussian_psf(128)
        seismic = seismic * noise
        
        # normalize the seismic data
        Nstds = random.randint(1,7)
        seismic_standardized = zscore(seismic, Nstds)
        
        # transpose inline and crossline with a random probability
        if np.random.uniform() > 0.5:
            seismic = seismic.transpose(0,2,1)
            fault = fault.transpose(0,2,1)
        
        # convert to torch tensor
        seismic_tensor = torch.from_numpy(seismic_standardized).float().unsqueeze(0)
        fault_tensor = torch.from_numpy(fault).float().unsqueeze(0)
        
        return seismic_tensor, fault_tensor


if __name__=='__main__':
    # file paths
    bucket = 'misc-bucket-new'
    seismic_path = 'synthetic-fault-data/seismic/seismic'
    fault_path = 'synthetic-fault-data/fault/fault'

    # create dataset and dataloader
    dataset = SyntheticSeismicWithGroundTruth(bucket, seismic_path, fault_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # sample and plot from the dataloader
    x, y = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    i = random.randint(0,127)
    plt.imshow(x.detach().cpu().numpy().squeeze()[i].T, cmap='gray', vmin=-1, vmax=1)
    plt.imshow(y.detach().cpu().numpy().squeeze()[i].T, cmap='Greys', vmin=0, vmax=1, alpha=0.5)
    plt.savefig('/home/ubuntu/projects/faultseg/figs/test_plot.png')