"""Script contains trainer classes for managing training of deep models for fault segmentation"""
from .dataloaders import SyntheticSeismicWithGroundTruth
from .models import unet_3D_DS
from .dice import DiceLoss
from torch.utils.data import DataLoader
import torch
import boto3
import torch.nn as nn
from .utils import *


class FaultEstimator3D():
    def __init__(self, args):
        """initializes fault estimator for training and/or inference for fault mapping"""
        
        self.bucket = args['bucket']
        
        # Extract training-relevant parameters
        if args['training']:
            self.seis_path = args['seismic_file_path']
            self.fault_path = args['fault_mask_path']
            self.epochs = args['epochs']
            self.alpha = args['alpha']  # weight for cross entropy loss
            self.gamma = args['gamma']  # weight for dice loss
            self.lr = args['learning_rate']  # learning rate
            self.pos_weight = args['pos_weight']  # positive weight for fault class voxels
            self.checkpoint_save_dir = args['checkpoint_save_dir']  # checkpoint dir for saving models and figs
            self.last_train_checkpoint = args['last_train_checkpoint']  # in case resuming training from an earlier checkpoint
            self.save_frequency = args['save_freq']  # checkpointing frequency in terms of number of epochs
        
        # extract inference relevant parameters
        if args['inference']: 
            self.test_checkpoint_path = args['test_checkpoint_path']
            self.test_vol_path = args['test_seismic_vol_path']
            self.chunk_size = args['chunk_size']
            self.overlap = args['overlap']
            self.clipping_std = args['clipping_std']
            self.test_checkpoint_save_dir = args['test_checkpoint_save_dir']
            self.threshold = args['threshold']
        
    def train(self):
        
        # set up trainloader
        dataset = SyntheticSeismicWithGroundTruth(self.bucket, self.seis_path, self.fault_path)
        trainloader = DataLoader(dataset, batch_size=1,shuffle=True, num_workers=3)
        
        # initialize model
        model = unet_3D_DS().cuda()
        
        # initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # loss function
        loss_fn1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]).cuda())
        loss_fn2 = DiceLoss()
        
        # resume from a previous checkpoint if provided
        if self.last_train_checkpoint is not None:
            saved_checkpoint = load_checkpoint_from_s3(self.bucket, self.last_train_checkpoint)
            model.load_state_dict(saved_checkpoint['model_state_dict'])
            optimizer.load_state_dict(saved_checkpoint['optimizer_state_dict'])
            start_epoch = saved_checkpoint['epoch']
        else:
            start_epoch = 0   
        
        try:
            for epoch in range(start_epoch, self.epochs):
                for i, (x, y) in enumerate(trainloader):
                    model.train()
                    optimizer.zero_grad()
                    
                    x, y = x.cuda(), y.cuda()
                    out = model(x)
                    loss = self.alpha*loss_fn1(out, y) + self.gamma*loss_fn2(out, y)
                    loss.backward()
                    optimizer.step()
                    
                    print('Epoch: {} | Iteration: {} | Train Loss: {:0.4f}'.format(epoch, i, loss.item()))
                    
                if epoch % self.save_frequency == 0:
                    save_path = self.checkpoint_save_dir + '/models/checkpoint_ep_{}.pt'.format(epoch)
                    save_checkpoint_to_s3(model, optimizer, epoch, self.bucket, save_path)
                    save_test_plots_to_s3(model, trainloader, self.bucket, self.checkpoint_save_dir+'/figures')
                    
            
        except KeyboardInterrupt:
            print('Ctrl+C detected. Saving checkpoint...')
            save_path = self.checkpoint_save_dir + '/models/checkpoint_ep_{}.pt'.format(epoch)
            save_checkpoint_to_s3(model, optimizer, epoch, self.bucket, save_path)
            save_test_plots_to_s3(model, trainloader, self.bucket, self.checkpoint_save_dir+'/figures')
                
                
    def predict(self):
        # load the volume
        seismic = load_numpy_array_from_s3(self.bucket, self.test_vol_path)
        seismic = seismic.transpose(2,0,1)[50:,:200,:400]  # transpose to bring depth to first dimension
        
        # initialize model
        model = unet_3D_DS().cuda()
        
        # load checkpoint
        test_checkpoint = load_checkpoint_from_s3(self.bucket, self.test_checkpoint_path)['model_state_dict']
        model.load_state_dict(test_checkpoint)
        
        # generate fault cube
        print('Starting Inference...')
        fault_vol = cubing_prediction(model, seismic, self.chunk_size, self.overlap, self.clipping_std)
        
        # grab inlines from fault volume and send to s3
        print('Generating Inline Predictions')
        for i in range(0, seismic.shape[1]-1, 25):
            fig = plt.figure()
            plt.imshow(seismic[:,i,:], cmap='gray', aspect='auto')
            plt.imshow(threshold(fault_vol[:,i,:], self.threshold), cmap='Reds', aspect='auto', alpha=0.5)
            save_fig_to_s3(fig, self.bucket, self.test_checkpoint_save_dir + '/inline_{}.png'.format(i))
            plt.close('all')
            
        # grab crosslines from fault volume and send to s3
        print('Generating Crossline Predictions')
        for i in range(0, seismic.shape[2]-1, 25):
            fig = plt.figure()
            plt.imshow(seismic[:,:,i], cmap='gray', aspect='auto')
            plt.imshow(threshold(fault_vol[:,:,i], self.threshold), cmap='Reds', aspect='auto', alpha=0.5)
            save_fig_to_s3(fig, self.bucket, self.test_checkpoint_save_dir + '/xline_{}.png'.format(i))
            plt.close('all')
        
        
        # grab depth slices from fault volume and send to s3
        print('Generating Depth Predictions')
        for i in range(0, seismic.shape[0]-1, 25):
            fig = plt.figure()
            plt.imshow(seismic[i], cmap='gray', aspect='auto')
            plt.imshow(threshold(fault_vol[i], self.threshold), cmap='Reds', aspect='auto', alpha=0.5)
            save_fig_to_s3(fig, self.bucket, self.test_checkpoint_save_dir + '/depth_{}.png'.format(i))
            plt.close('all')
            
        
        
        