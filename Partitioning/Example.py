# Just an example script using CIFAR10
import torch, torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import Encoders, Methods, DataAug, Scheduler, Utils
from Partitioning.ModelFlattening import FlattenResnet #Assuming you imported everything and are in the parent directory
from Partitioning.PartitionAwareMethods import GPipeSimCLR # Partitioning aware methods objects, the rest are for single GPU or distributed data parallel only
from torchgpipe import GPipe # For model and pipeline parallelism

# Some Important variables
BATCH_SIZE      = 30
EPOCHS          = 50
SAVEPATH        = ''
IMAGEPATH       = ''

# Instantiate an encoder
def resnet50(outputchannels=1024, modification_type={''}):
  
  return Encoders.ResNet(
      block = Encoders.Bottleneck, 
      layers = [3, 4, 6, 3], 
      outputchannels=outputchannels, 
      modification_type={
                'resnetB', 
                'resnetC',
                'resnetD',
                'resnext', 
                'squeezeandexcite',
                'stochastic_depth', # deleted preactivation residual unit although I think it is still a useful modification
                },
      
      )
  
encoder = resnet50()

# Instantiate the dataloader
custom_transforms = DataAug.ContrastiveTransformations(
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

# Replace the following line if you are going to train off of an imagefolder
trainset         = torchvision.datasets.ImageFolder(
                    root          =   IMAGEPATH, # add your image path here
                    transform     =   contrast_transforms,
                    )
dataloader      = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# Assuming the use of a CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # This line is not needed, just here for illustration

# Prepare encoder's pipeline parallelism strategy using GPipe
flattened_resnet_model = FlattenResnet(encoder) # Flatten it to sequential neural network first
# We will now upload the model to GPipe
encoder_gpipe = GPipe(flattened_resnet_model,
              balance=[3,3,3,3,3,3],  # Specify GPUs.
              devices = [0,0,0,0,0,0],
              chunks=3,
              checkpoint='never',#'always',#'never'
      )

# Instantiate the SSL model with input as the GPipe model
model     = GPipeSimCLR( # The only difference here is it knows, based off encoder_gpipe.devices, which GPU needs images, and which GPU calculates loss
          encoder         = encoder_gpipe, 
          batch_size      = BATCH_SIZE, 
          epochs          = EPOCHS,
          savepath        = SAVEPATH,
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

