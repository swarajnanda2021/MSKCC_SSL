
# Self-supervised learning library for computer vision
A suite of self-supervised learning techniques and image encoders are provided in this repo, written in PyTorch with ample help from other repos (cited in code), papers, and GPT4. The methods were tested in google colab, and on the [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [imagewoof](https://github.com/fastai/imagenette) datasets. No performance parameters are provided as the goal of creating this repo was for personal education purposes only.

## Introduction

As preparation for my upcoming postdoctoral venture at MSKCC, or the Memorial Sloan Kettering Cancer Center, I spent a few months studying and implementing various image encoder learning techniques that would leverage the vast amount of image data available at MSKCC upon my arrival. In the following, I will describe the contents in this repo. I have written the code so far for full precision, and considering only a single GPU, as I did not have a precision sensitive GPU like a V100, or multiple GPUs during my study period.

## Contents

- **A.** [Methods](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Methods.py): A selection of self supervised learning techniques presented as torch.nn.Module objects, which are _mostly_ self-contained apart from files in [Utils](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Utils.py). A special focus is placed on joint-embedding architectures (see SSL cookbook: https://arxiv.org/abs/2304.12210), although the [Masked Autoencoder](https://arxiv.org/abs/2111.06377) is also provided, more of a personal curiosity, but I think the future is joint-embedding based because it is cheaper (you are calculating losses only in the embedding state, not in the dimension of the input that is decoded from the embedding state).

- **B.** [Encoders](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Encoders.py): A selection of image encoder objects, which are not instantiated. You will need to instantiate it using the help provided subsequently in this readme. We have so far in our selection the basic ResNet and Vision Transformer. Instantiation can also be sought from at the bottom end of the code. For ResNet, we have provided all modifications up to ResNeXT, and for transformers, we have provided the Relative Attention module so that you may consider it a building block for hybrid architectures.

- **C.** [Schedulers](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Schedulers.py): So far, I only have in this module the one found in [Katsura's github repo](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py). This suits the purpose for now, but **other schedulers are coming soon!**

- **D.** [Utils](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Utils.py): I have here the utilities needed for some of the self-supervised learning techniques, such as the **MultiCropWrapper** and **DinoProjection** head used in [DiNOv1](https://arxiv.org/abs/2104.14294), **SupportSet** for the [NNCLR](https://arxiv.org/abs/2104.14548) method, and **LARS** optimizer copied verbatim from Meta's [DiNO utils](https://github.com/facebookresearch/dino/blob/main/utils.py).

- **E.** [DataAug](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/DataAug.py): Data augmentation techniques, classical (linear transformations to the image), are found here. There are two kinds of data augmentation:
   - **E.1.** Contrastive learning based data augmentation: Outputs a set of 2 (or n) views of transformed input image batch using classical techniques
   - **E.2.** DiNO data augmentation technique. Here, there is a local and a global crop needed for the teacher-student learning approach found in DiNO. Classical augmentation is applied to all crops.
 
There are other elements in the repo, such as [DimensionReduction](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/DimensionReduction.py), which basically is my implementation of the [UMAP](https://arxiv.org/abs/1802.03426) method, from scratch, but it is slow. I recommend using a faster library. Other elements also include the training script [Training](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Training.py), but that only serves as a means to understand how to combine these modules to your data. There is also a [jupyter notebook](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Squeeze_and_attend_mamba.ipynb) which is basically just a little test I was doing on an architecture I call the 'Squeeze and Attend' method combined with a [Mamba](https://arxiv.org/abs/2312.00752) block that I wrote from scratch. Won't go into details, it isn't very useful right now. The Mamba block is slow as it uses the recurrent version of the selective scan, so it won't scale either. I use a standard convolution block prior to the state-space model block in the Mamba block, but that should be replaced with a CUDA optimized causal-conv1d method ([example](https://github.com/Dao-AILab/causal-conv1d)).

## Usage

### Instantiation of Encoder Object

I will describe here only the approach for instantiating a resnet object and a vision transformer object. The vision transformer is fairly vanilla in implementation but there are several modifications acceptable by the resnet instantiation approach. 

- **ResNet** : Here, I have instantiated the ResNet method, which takes a typical resnet block. The _BasicBlock_ has no bottleneck structure, whereas the _Bottleneck_ block has.
   ```ruby
   import Encoders
   from Encoders import ResNet, BasicBlock, Bottleneck
   
   def resnet34(outputchannels=1000, modification_type={''}):
       
       return ResNet(
           block = BasicBlock,
           layers = [3, 5, 7, 5], 
           outputchannels=outputchannels, 
           modification_type={
                     'resnetB', 
                     'resnetC',
                     'resnetD',
                     'resnext', 
                     'squeezeandexcite',
                     'stochastic_depth',
                     }
           )
   model = resnet34()
   ```
   Here, the input modification_type refers to modifications that are possible to the general vanilla ResNet from the 2015 [He et al. paper](https://arxiv.org/abs/1512.03385). These are namely:
  - Resnet B, C, and D: Based on the [Bag of Tricks](https://arxiv.org/abs/1812.01187) paper that discusses several modifications to both the input stem and the resnet blocks.
  - Squeeze and Excite: Makes representations more expressive by condensing the representation size by 1/4th of original, and then applying a sigmoid activation before re-expanding it. Based on the [Squeeze and Excite](https://arxiv.org/abs/1709.01507) paper.
  - Stochastic Depth  : A regularization technique that drops out blocks based on a survival probability (you'll need to manually tweak this in the Encoders code), so that the model does not overfit. Necessary when data size is small and/or model size is large. Use strong values for smaller datasets or larger models, and relax this criteria when the dataset size is large. It is difficult to tell for your application what large and small mean, think in millions of images for large. Based on the [Stochastic depth](https://arxiv.org/abs/1603.09382) paper.
 
   Furthermore, there are several ResNets that can be instantiated. The ResNet-18 and ResNet-34 models can be instantiated using the Encoders.BasicBlock method, and following a layer depth structure (number of blocks per stage in the ResNet, of which there are 4). Here are some possibilities:
   - Resnet-18 :
     - layers=[2, 2, 2, 2]
     - block = BasicBlock
   - Resnet-34 :
     - layers=[3, 4, 6, 3]
     - block = BasicBlock
   - Resnet-18 :
     - layers=[3, 4, 6, 3]
     - block = Bottleneck
  
- **Vision Transformer** : The following is how you would instantiate the vision transformer. But mind you, the attention mechanism is quite useless in the low data-regime due to a lack of inductive bias in the attention operation.
  ```ruby
  def ViT_base():
        model = ViT_encoder(image_size = 32, 
                    patch_size = 16, 
                    in_channels=3, 
                    embedding_dim = 768, 
                    feature_size= 1000,
                    n_blocks = 12,
                    n_heads = 12,
                    mlp_ratio = 4.0,
                    qkv_bias = True,
                    attention_dropout = 0.2,
                    projection_dropout = 0.2)
        return model
    
  def ViT_tiny():
        model = ViT_encoder(image_size=32, 
                            patch_size=16, 
                            in_channels=3, 
                            embedding_dim=192,  # Reduced embedding dimension
                            feature_size=1000,
                            n_blocks=12,        # Similar to the base model
                            n_heads=3,          # Fewer heads than the base model
                            mlp_ratio=4.0,
                            qkv_bias=True,
                            attention_dropout=0.1,  # You might adjust dropout rates
                            projection_dropout=0.1)
        return model


  def ViT_small():
        model = ViT_encoder(image_size=32, 
                            patch_size=16, 
                            in_channels=3, 
                            embedding_dim=384,  # Increased compared to tiny
                            feature_size=1000,
                            n_blocks=12,        # Similar to the base model
                            n_heads=6,          # More heads than tiny, fewer than base
                            mlp_ratio=4.0,
                            qkv_bias=True,
                            attention_dropout=0.1,
                            projection_dropout=0.1)
        return model
  ```
   

### Instantiation of Data Augmentation Object

- **Contrastive** : I will take here the example of producing two views of a batch of images from the CIFAR dataset in order to instantiate the data augmentation pipeline. The entries are fairly self-explanatory so I do not need to describe them in detail. 
   ```ruby
   import torch, torchvision
   from torchvision.datasets import CIFAR10
   from DataAug import ContrastiveTransformations
   from torch.utils.data import DataLoader
   
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
   trainset        = CIFAR10(root='./data',train=True,download=True, transform=contrastive_transform)
   dataloader      = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
   ```

- **DiNO**: Here, we will take the case of 2 global crops of size 224x224 pixels (basically set as a constant in the DataAug method which you can change manually if you would like), and then 6 local crops of size 96x96 pixels. The scale of these local and global crops are a factor of the image dimensions, and the values specified in the tuple is (low end range, high end range) in the instantiation of the DinoTransforms object.

   ```ruby
   import torch, torchvision
   from torchvision.datasets import CIFAR10
   from DataAug import DinoTransforms
   from torch.utils.data import DataLoader
   
   CROPS           = 6
   BATCH_SIZE      = 256
   trainset        = torchvision.datasets.ImageFolder(                                            # Used this because CIFAR10 is too small for doing this kind of SSL (IMO)
                       root          =   './drive/MyDrive/SSL_Datasets/imagewoof2-320/train',    # add your image path here
                       transform     =   DinoTransforms(
                                     local_size         = 96,
                                     global_size        = 224,
                                     local_crop_scale   = (0.05, 0.4),
                                     global_crop_scale  = (0.4, 1.0),
                                     n_local_crops      = CROPS,
                                     )
                       )
   dataloader       = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
   ```

### Instantiation of a Self-Supervised Learning Method Object

### Training on your data











