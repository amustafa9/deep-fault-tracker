import sys
import os

from models import *
from utils import *
import torch
from io import BytesIO
import boto3


model = unet_3D_DS()
# bucket = 'misc-bucket-new'
# checkpoint_path = 'fault-training-checkpoints/models/checkpoint_ep_187.pt'
# save_path = 'unet_torchscript_128.pt'
# checkpoint = load_checkpoint_from_s3(bucket, checkpoint_path)['model_state_dict']
# model.load_state_dict(checkpoint)
# traced_model = torch.jit.trace(model, (torch.randn(1,1,128,128,128),))
# # Create a BytesIO buffer to store the checkpoint
# buffer = BytesIO()
# torch.save(traced_model, buffer)
# buffer.seek(0)

# # Upload the checkpoint to S3
# s3 = boto3.client('s3')
# s3.upload_fileobj(buffer, bucket, save_path)