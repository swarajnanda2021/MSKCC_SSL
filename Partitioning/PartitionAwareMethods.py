import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoders import PatchEmbed, TransformerBlock
from Utils import SupportSet, MultiCropWrapper, DiNOProjection
import numpy as np

# Modify simCLR to handle an encoder that has been partitioned and fed to the torchgpipe module
# The main modification is that the encoder that is provided at simCLR's init should have a 'devices'
# property such that encoder.devices[0] is where the images are fed in, and encoder.devices[-1] is where 
# the loss is calculated.
class GPipeSimCLR(nn.Module): # the similarity loss of simCLR

    def __init__(self, encoder, batch_size,epochs,savepath):
        super().__init__()
        self.model = encoder # define the encoder here, passed through GPipe, so in GPU already
        self.device_in, self.device_out = encoder.devices[0], encoder.devices[-1]
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device_out)
        self.batch_size = batch_size
        self.epochs = epochs
        self.savepath  = savepath
        self.losses = []  # Track losses

    def SimCLR_loss(self, features):
        n_views = 2
        # Define the similarity matrix's pattern (which elements are related in the batch and which aren't)
        #original_tensor = torch.arange(0,self.batch_size,1)
        # Create the repeated pattern
        #pattern = torch.repeat_interleave(original_tensor, repeats=n_views)
        pattern = torch.cat([torch.arange(self.batch_size) for i in range(n_views)], dim=0)
        # make similarity matrix by the above method (need to understand this)
        pattern = (pattern.unsqueeze(0) == pattern.unsqueeze(1)).float()
        pattern = pattern.to(self.device_out)
        mask = torch.eye(pattern.shape[0])
        mask = mask.to(self.device_out)
        pattern = pattern-mask

        features = F.normalize(features,dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix - mask

        # select and combine positives
        positives = similarity_matrix[pattern.bool()].view(pattern.shape[0],-1)
        negatives = similarity_matrix[~pattern.bool()].view(similarity_matrix.shape[0],-1)

        logits = torch.cat([positives,negatives],dim=1)
        # we have to further develop the logits function as follows: we will define a temperature argument that sets the shape of the distribution
        temperature = 0.07
        logits = logits / temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device_out)

        loss = self.criterion(logits, labels)
        return loss

    def get_encoder(self):
        return self.model

    def save_checkpoint(self, file_path, epoch, optimizer, scheduler):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
            'losses': self.losses,  # Save losses
            'epoch': epoch
        }
        torch.save(checkpoint, f"{file_path}_epoch_{epoch}.pth")
        print(f"Checkpoint saved to {file_path}_epoch_{epoch}.pth")

    def load_checkpoint(self, file_path, device, optimizer, scheduler):
        checkpoint = torch.load(file_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:  # Check if the checkpoint includes a scheduler state
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'losses' in checkpoint:  # Load losses if available
            self.losses = checkpoint['losses']
        print(f"Checkpoint loaded from {file_path}")



    def train(self, dataloader, scheduler, optimizer):
        

        # Start training
        for epoch in range(self.epochs):
            # Initialize tqdm progress bar
            train_loader = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for views, _ in train_loader:  # Unpack data and labels from each batch
                if views[0].size(0) != self.batch_size:
                    continue  # Skip this batch
                imgs = torch.cat(views, dim=0).to(self.device_in, non_blocking=True)
                    
                # Load images and calculate InfoNCE loss
                features = self.model(imgs)
                loss = self.SimCLR_loss(features)
                #labels = logits.to(self.device_out, non_blocking=True)
                #loss = self.criterion(logits, labels) # both are automatically in 

                # Append the loss
                self.losses.append(loss.item())

                # Perform optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update tqdm progress bar with the current loss and learning rate
                train_loader.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            if (int(epoch)%10 == 0):
              file_path = self.savepath

              # Save the current state of the model and optimizer
              self.save_checkpoint(file_path, epoch, optimizer, scheduler)


            scheduler.step(epoch)


        return self.losses

    

