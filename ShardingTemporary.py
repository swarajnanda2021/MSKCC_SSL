

# This is the simCLR I have to rewrite to allow sharding:

# Modify simCLR to handle encoder_gpiped

class gpipe_simCLR(nn.Module): # the similarity loss of simCLR

    def __init__(self, encoder, batch_size,epochs,savepath):
        super().__init__()
        self.model = encoder # define the encoder here, passed through GPipe, so in GPU already
        self.device_in, self.device_out = 'cuda:0', 'cuda:0'
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

    
# The following code flattens the Encoders.ResNet in order to make it easier for GPipe to digest:
def flatten_resnet_for_gpipe(resnet_model):
    modules = []

    # Input block
    modules.append(nn.Sequential(*[resnet_model.input_branch]))

    # Process each layer's blocks, where the first block is usually the downsampling one
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(resnet_model, layer_name)
        for i, bottleneck in enumerate(layer):
            # Each bottleneck block as a separate sequential
            modules.append(nn.Sequential(bottleneck))

    # Final stage combined into one block
    final_stage = nn.Sequential(
        resnet_model.avgpool,
        nn.Flatten(1),
        resnet_model.fc
    )
    modules.append(final_stage)

    return nn.Sequential(*modules)

# Assuming 'resnet_model' is your original ResNet model
flattened_resnet_model = flatten_resnet_for_gpipe(encoder)


# We will now upload the model to GPipe
from torchgpipe import GPipe
encoder_gpipe = GPipe(flattened_resnet_model,
              balance=[9,9],  # Specify GPUs.
              devices = [0,0],
              chunks=1,
              checkpoint='never')

custom_transforms = ContrastiveTransformations(
            size=32,
            nviews=2,
            horizontal_flip=True,
            resized_crop=True,
            color_jitter=True,
            random_grayscale=True,
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.1,
            color_jitter_p=0.8,
            grayscale_p=0.2,
            to_tensor=True,
            normalize=True,
            mean=(0.5,),
            std=(0.5,)
        )

BATCH_SIZE      = 256
EPOCHS          = 50

trainset        = CIFAR10(root='./data',train=True,download=True, transform=custom_transforms)
dataloader      = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# Assuming the use of a CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model     = gpipe_simCLR(
          encoder         = encoder_gpipe, 
          #device          = device, #uncomment if Methods.simCLR is used
          batch_size      = BATCH_SIZE, 
          epochs          = EPOCHS,
          savepath        = './test.pth',
      )

optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.05) 
scheduler  = Scheduler.CosineAnnealingWarmupRestarts(
                        optimizer,
                        first_cycle_steps=EPOCHS - 10,  # Total epochs minus warm-up epochs
                        cycle_mult=1.0,  # Keep cycle length constant after each restart
                        max_lr=1e-3,  # Maximum LR after warm-up
                        min_lr=1e-5,  # Minimum LR
                        warmup_steps=10,  # Warm-up for 10 epochs
                        gamma=1.0  # Keep max_lr constant after each cycle
                    )

loss_iter  = model.train(dataloader = dataloader, scheduler = scheduler, optimizer = optimizer)








