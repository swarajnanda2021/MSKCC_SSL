from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import DataAug, Encoders, Methods
import torch
from Scheduler import CustomScheduler

BATCH_SIZE = 1024
EPOCHS = 50



contrast_transforms = DataAug.ContrastiveTransformations(
        size            =   32,
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
cifar_trainset   = CIFAR10(root='./data',train=True,download=True, transform=contrast_transforms) # please do not use this.
trainset         = torchvision.datasets.ImageFolder(
                    root          =   './drive/MyDrive/SSL_Datasets/imagewoof2-320/train', # add your image path here
                    transform     =   contrast_transforms,
                    )

train_loader    = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize encoder and simCLR model
#encoder = Encoder(4000)
def resnet34():
    layers   = [3, 5, 7, 5] # [2 2 2 2] for resnet18()
    model    = Encoders.ResNet(Encoders.BasicBlock, layers,1000)
    return model

def resnet50():# 24.557120M parameters
        layers=[3, 4, 6, 3]
        return Encoders.ResNet(block = Encoders.Bottleneck, layers = layers, outputchannels = 512)


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

encoder  = resnet50()#ViTencoder() #resnet34()
device   = torch.device("cuda")


simclr_model    = Methods.simCLR(
                            encoder         = resnet34(), 
                            device          = device, 
                            batch_size      = BATCH_SIZE, 
                            epochs          = EPOCHS,
                            savepath        = './checkpoints/test.pth',
                            )

nnclr_model     = Methods.NNCLR(
                            encoder = resnet34(),
                            feature_size                    = 512, 
                            queue_size                      = 32768,
                            projection_hidden_size_ratio    = 4,
                            prediction_hidden_size_ratio    = 4, 
                            temperature                     = 0.1, 
                            reduction                       = 'mean',
                            batch_size                      = BATCH_SIZE, 
                            epochs                          = EPOCHS,
                            device                          = device,
                            savepath        = './checkpoints/test.pth',
                            )
                            

mae_model       = Methods.MAE(
                                image_size              = 32,            # image_size (int)              : size of the input image
                                patch_size              = 8,            # patch_size (int)              : size of the patches to be extracted from the input image
                                in_channels             = 3,           # in_channels (int)             : number of input channels
                                embedding_dim           = 512,         # embedding_dim (int)           : number of elements of the embedding vector (per patch)
                                feature_size            = 1024,          # feature_size (int)            : Total size of feature vector
                                n_blocks                = 8,              # n_blocks (int)                : total number of sequential transformer blocks (a.k.a. depth)
                                n_heads                 = 8,               # n_heads (int)                 : total number of attention heads per transformer block
                                mlp_ratio               = 4,             # mlp_ratio (float)             : the ratio by which embedding dimension expands inside a transformer block (in the MLP layer after attention)
                                qkv_bias                = True,              # qkv_bias (bool)               : whether to add a bias term to the qkv projection layer or not
                                attention_dropout       = 0.5,     # attention_dropout (float)     : dropout in the attention layer
                                projection_dropout      = 0.5,    # projection_dropout (float)    : dropout in the projection layer
                                mask_ratio              = 0.75,            # mask_ratio (float)            : masking applied to input image patch embeddings
                                decoder_embedding_dim   = 512, # decoder embedding dim (float) : decoder has a different embedding dim for convenience
                                decoder_n_heads         = 8,
                                decoder_n_blocks        = 8,
                                epochs                  = EPOCHS,
                                device                  = device,
                                batch_size              = BATCH_SIZE

                )


# The following are some settings I needed to begin the DiNO training.

# I need this because we have to blank out the last fully connected layers of our encoder, and thus I need to understand the final 
# size of the representations prior to the projection. Of course, you'll replace this with a fully connected layer while freezing the
# entire pre-trained encoder using torch no_grad.
def get_encoder_output_dim(encoder, input_size=(1, 3, 224, 224)): 
    # Create a dummy input tensor of the specified size
    encoder.head, encoder.fc = nn.Identity(), nn.Identity()
    dummy_input = torch.randn(input_size)

    # Run a forward pass through the encoder
    with torch.no_grad():
        encoder.eval()
        output = encoder(dummy_input)

    # Return the size of the output tensor (excluding the batch dimension)
    return output.size(1)

BATCH_SIZE = 80 # for the images I have, this filled up my GPU upto 14.7 GB.
EPOCHS     = 50
CROPS      = 6
EMBEDDING_DIM = 512

trainset_dino = torchvision.datasets.ImageFolder(
                    root          =   './drive/MyDrive/SSL_Datasets/imagewoof2-320/train', # add your image path here
                    transform     =   DataAug.DinoTransforms(
                                  local_size         = 96,
                                  global_size        = 224,
                                  local_crop_scale   = (0.05, 0.4),
                                  global_crop_scale  = (0.4, 1.0),
                                  n_local_crops      = CROPS,
                                  )
                    )
dataloader_dino = DataLoader(trainset_dino, batch_size=BATCH_SIZE, shuffle=True)


encoder = resnet34()#ViT_tiny()#resnet50()#resnet18()
encoder_output_dim = get_encoder_output_dim(encoder, input_size=(1, 3, 224, 224))
print("Output dimension of encoder:", encoder_output_dim)

dino_model      = Methods.DiNO(
            encoder_embedding_dim = encoder_output_dim,#encoder_output_dim, based only on the global crop size.
            feature_size          = 128,
            encoder               = encoder,
            device                = device,
            batch_size            = BATCH_SIZE,  # Adjust as needed
            epochs                = EPOCHS,  # Adjust as needed
            temperature_teacher   = 0.04,  # Example value, adjust as needed
            temperature_student   = 0.07,  # Example value, adjust as needed
            ncrops                = CROPS,
            savepath              = './checkpoints/test.pth',
        )


BYOL_model = BYOL(
                encoder     = encoder,
                device      = device,
                savepath   = './drive/MyDrive/SSL_Models/BYOL_model.pth',
                batch_size  = BATCH_SIZE,
                epochs      = 50,
                feature_size = FEATURE_SIZE,
                projection_hidden_size_ratio = 2,
                prediction_hidden_size_ratio = 2,
                alpha       = 0.96
          )


barlow_twins = BarlowTwins(encoder=encoder,
                           device=device,
                           batch_size=BATCH_SIZE,
                           epochs=50,
                           savepath='./drive/MyDrive/SSL_Models/barlowtwins_checkpoint.pth',
                           feature_size=FEATURE_SIZE, # important: a large projector output size often improves performance. It might be worthwhile to have this implemented.
                           projection_hidden_size_ratio=2,
                           gamma=0.0051)


vicreg_model = VICReg(encoder=encoder,
                      device=device,
                      epochs=100,
                      savepath='./drive/MyDrive/SSL_Models/VICREG_checkpoint.pth',
                      batch_size=BATCH_SIZE,
                      feature_size=FEATURE_SIZE,
                      projection_hidden_size_ratio=2,
                      projector_num_layers=2,
                      output_projector_size=FEATURE_SIZE*2*2)


if __name__ == "__main__":
      
      simclr_loss_iter               =   simclr_model.train(train_loader)
      #dino_loss_iter                =   dino_model.train(dataloader_dino) 
      #vicreg_loss_iter              =   vicreg_model.train(train_loader)  
      #nnclr_loss_iter               =   nnclr_model.train(train_loader)
      #mae_loss_iter                 =   mae_model.train(train_loader)  
      #byol_loss_iter                =   BYOL_model.train(train_loader)
      #barlowtwins_loss_iter         =   barlow_twins.train(train_loader)

