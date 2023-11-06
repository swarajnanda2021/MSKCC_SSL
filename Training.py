from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import DataAug, Encoders, Methods
import torch

BATCH_SIZE = 1024
EPOCHS = 20


contrast_transforms = DataAug.ContrastiveTransformations(
        nviews          =   2,
        horizontal_flip =   True,
        resized_crop    =   True,
        color_jitter    =   True,
        random_grayscale=   True,
        brightness      =   0.5,
        contrast        =   0.5,
        saturation      =   0.5,
        hue             =   0.1,
        color_jitter_p  =   0.8,
        grayscale_p     =   0.2,
        to_tensor       =   True,
        normalize       =   True,
        mean            =   (0.5,),
        std             =   (0.5,)
    )

# Initialize dataset and dataloader
cifar_trainset  = CIFAR10(root='./data',train=True,download=True, transform=contrast_transforms)
train_loader    = DataLoader(cifar_trainset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize encoder and simCLR model
#encoder = Encoder(4000)
def resnet34():
    layers   = [3, 5, 7, 5]
    model    = Encoders.ResNet(Encoders.ResidualBlock, layers,1000)
    return model

def ViTencoder():
        model = Encoders.ViT_encoder(
                    image_size          =   32, 
                    patch_size          =   16, 
                    in_channels         =   3, 
                    embedding_dim       =   512, 
                    feature_size        =   1000,
                    n_blocks            =   12,
                    n_heads             =   8,
                    mlp_ratio           =   4.0,
                    qkv_bias            =   True,
                    attention_dropout   =   0.2,
                    projection_dropout  =   0.2)
        return model

encoder  = ViTencoder() #resnet34()
device   = torch.device("cuda")

simclr_model    = Methods.simCLR(
                            encoder         = resnet34(), 
                            device          = device, 
                            batch_size      = BATCH_SIZE, 
                            epochs          = EPOCHS)

nnclr_model     = Methods.NNCLR(
                            feature_size                    = 512, 
                            queue_size                      = 32768,
                            projection_hidden_size_ratio    = 4,
                            prediction_hidden_size_ratio    = 4, 
                            temperature                     = 0.1, 
                            reduction                       = 'mean',
                            batch_size                      = BATCH_SIZE, 
                            epochs                          = EPOCHS,
                            device                          = device
                            )


if __name__ == "__main__":
      
      #simclr_loss_iter  =   simclr_model.train(train_loader)
      nnclr_loss_iter   =   nnclr_model.train(train_loader)


