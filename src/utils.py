import boto3
import io
import s3fs
import numpy as np
import torch
from io import BytesIO
import matplotlib.pyplot as plt
import random

def list_s3_files(bucket_name, prefix="", extension=""):
    """
    List all files in a specific S3 directory with a given extension.

    :param bucket_name: Name of the S3 bucket.
    :param prefix: Directory path inside the bucket. (e.g., 'my-folder/')
    :param extension: File extension to filter by (e.g., '.txt', '.csv').
    :return: List of file keys matching the extension.
    """
    s3_client = boto3.client('s3')
    file_keys = []

    # Use paginator to handle large lists of files
    paginator = s3_client.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix}

    for page in paginator.paginate(**operation_parameters):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith(extension):  # Filter by extension
                    file_keys.append(key)

    return file_keys

def get_cube_from_bytestream(bucket, filepath):
    """function returns a seismic or fault cube give the file path to the binary file in an s3 bucket"""
    s3_client = boto3.client('s3')
    byte_obj = s3_client.get_object(Bucket=bucket, Key=filepath)
    byte_stream = byte_obj['Body'].read()
    data = np.frombuffer(byte_stream, dtype=np.single).reshape(128,128,128)
    return data

def zscore(data, Nstds=3):
    """scales and normalizes input data to be centered between -1 and 1"""
    mu, sigma = data.mean(), data.std()
    data_centered = data - mu
    data_clipped = np.clip(data_centered, a_min=-Nstds*sigma, a_max=Nstds*sigma)
    data_standardized = data_clipped/(Nstds*sigma)
    return data_standardized


def save_checkpoint_to_s3(model, optimizer, epoch, s3_bucket, s3_key):
    """
    Saves a training checkpoint to an S3 bucket.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer used during training.
        epoch: The current epoch number.
        s3_bucket: The name of the S3 bucket.
        s3_key: The key (path) within the S3 bucket where the checkpoint will be saved.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Create a BytesIO buffer to store the checkpoint
    buffer = BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    # Upload the checkpoint to S3
    s3 = boto3.client('s3')
    s3.upload_fileobj(buffer, s3_bucket, s3_key)

    

def load_checkpoint_from_s3(s3_bucket, s3_key):
    """
    Loads a training checkpoint from an S3 bucket.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer used during training.
        s3_bucket: The name of the S3 bucket.
        s3_key: The key (path) within the S3 bucket where the checkpoint is stored.

    Returns:
        The epoch number loaded from the checkpoint.
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)

    # Create a BytesIO buffer to store the downloaded checkpoint
    buffer = BytesIO(obj['Body'].read())

    # Load the checkpoint from the buffer
    checkpoint = torch.load(buffer)

    # Load model and optimizer states
    checkpoint = {
        'epoch': checkpoint['epoch'],
        'model_state_dict': checkpoint['model_state_dict'],
        'optimizer_state_dict': checkpoint['optimizer_state_dict']
    }

    return checkpoint


def save_test_plots_to_s3(model, dataloader, bucket_name, fig_save_dir):
    """function uses provided model with dataloader to generate 
    sample predictions to be saved as figures to s3"""
    
    model.eval()
    with torch.no_grad():
        for i in range(5):
            x, y = next(iter(dataloader))  # sample some data
            x, y = x[[0]].cuda(), y[[0]].cuda()  # grab only first sample from batch
            out = model(x)
            slice = random.randint(0, x.shape[3]-1)
            plt.imshow(x.detach().cpu().numpy().squeeze()[:,slice,:], cmap='gray')
            plt.imshow(out.detach().cpu().numpy().squeeze()[:,slice,:], cmap='Reds', alpha=0.5)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Upload the plot to S3 
            s3 = boto3.client('s3') 
            file_name = fig_save_dir + '/fig_{}.png'.format(i)
            s3.upload_fileobj(buffer, bucket_name, file_name, ExtraArgs={'ContentType':'image/png'})
            

def sigmoid(data):
    return 1/(1+np.exp(-data))

def cubing_prediction(model, data, infer_size, ol, Nstds):
    with torch.no_grad():
        model.eval()
        n1, n2, n3 = infer_size, infer_size, infer_size
        m1, m2, m3 = data.shape
        c1 = np.ceil((m1 + ol) / (n1 - ol)).astype(int)
        c2 = np.ceil((m2 + ol) / (n2 - ol)).astype(int)
        c3 = np.ceil((m3 + ol) / (n3 - ol)).astype(int)
        p1 = (n1 - ol) * c1 + ol
        p2 = (n2 - ol) * c2 + ol
        p3 = (n3 - ol) * c3 + ol
        gp = np.zeros((p1, p2, p3)) + 0.5
        gy = np.zeros((p1, p2, p3), dtype=np.single)
        gp[:m1, :m2, :m3] = data

        # generate coordinate triplets
        C1, C2, C3 = np.meshgrid(np.arange(c1), np.arange(c2), np.arange(c3), indexing='ij')
        coords = np.concatenate((C1.reshape(-1,1), C2.reshape(-1,1), C3.reshape(-1,1)), axis=1)

        for coord in coords:
            k1, k2, k3 = coord[0], coord[1], coord[2]
            b1 = k1 * n1 - k1 * ol
            e1 = b1 + n1
            b2 = k2 * n2 - k2 * ol
            e2 = b2 + n2
            b3 = k3 * n3 - k3 * ol
            e3 = b3 + n3
            gs = torch.tensor(zscore(gp[b1:e1, b2:e2, b3:e3], Nstds), dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()
            Y = model(gs).detach().cpu().numpy()
            gy[b1:e1, b2:e2, b3:e3] = gy[b1:e1, b2:e2, b3:e3] + Y[0, 0, :, :, :]
    return sigmoid(gy[:m1, :m2, :m3])


def load_numpy_array_from_s3(bucket, key):
    """loads numpy array from s3 to memory"""
    # Initialize a session using Amazon S3
    s3 = boto3.client('s3')

    # Retrieve the object from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    array_data = response['Body'].read()

    # Load the NumPy array from the byte stream
    array = np.load(BytesIO(array_data))
    return array

def save_fig_to_s3(fig, bucket, key):
    """saves a matplotlib figure object to s3"""
    
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Upload the plot to S3 
    s3 = boto3.client('s3') 
    s3.upload_fileobj(buffer, bucket, key, ExtraArgs={'ContentType':'image/png'})
    
    
def threshold(data, thresh=0.8):
    """thresholds probability scores contained in array data"""
    mask = data <= thresh
    data[mask] = 0
    data[~mask] = 1
    return data


def generate_3d_gaussian_psf(cube_size, seed=None):
    """
    Generate a 3D Gaussian point spread function (PSF) in a cube.
    
    Args:
        cube_size (int): The size of the cube (assumes a cube of shape (cube_size, cube_size, cube_size)).
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        numpy.ndarray: A 3D numpy array containing the Gaussian PSF.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define the grid for the cube
    x = np.linspace(-1, 1, cube_size)
    y = np.linspace(-1, 1, cube_size)
    z = np.linspace(-1, 1, cube_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = np.stack([X, Y, Z], axis=-1)  # Shape: (cube_size, cube_size, cube_size, 3)
    
    # Random mean within the normalized cube
    mean = np.random.uniform(-0.5, 0.5, size=3)  # Mean is within a subregion of the cube
    
    # Random covariance matrix (positive definite)
    random_matrix = np.random.uniform(0.1, 0.5, size=(3, 3))
    covariance = random_matrix @ random_matrix.T  # Ensure positive-definite covariance
    
    # Compute the Gaussian PSF
    grid_flat = grid.reshape(-1, 3)  # Flatten grid for vectorized computation
    diff = grid_flat - mean  # Difference from the mean
    inv_covariance = np.linalg.inv(covariance)
    mahalanobis = np.sum(diff @ inv_covariance * diff, axis=1)  # Mahalanobis distance squared
    gaussian_psf_flat = np.exp(-0.5 * mahalanobis)  # Gaussian function
    gaussian_psf = gaussian_psf_flat.reshape(cube_size, cube_size, cube_size)
    
    # Normalize the PSF
    gaussian_psf /= np.sum(gaussian_psf)
    
    return 1 - gaussian_psf
