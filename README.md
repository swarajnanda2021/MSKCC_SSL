
# Self-supervised learning library for computer vision
A suite of self-supervised learning techniques and image encoders are provided in this repo, written in PyTorch with ample help from other repos (cited in code), papers, and GPT4. The methods were tested in google colab, and on the [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [imagewoof](https://github.com/fastai/imagenette) datasets. No performance parameters are provided as the goal of creating this repo was for personal education purposes only.

## Introduction

As preparation for my upcoming postdoctoral venture at MSKCC, or the Memorial Sloan Kettering Cancer Center, I spent a few months studying and implementing various image encoder learning techniques that would leverage the vast amount of image data available at MSKCC upon my arrival. In the following, I will describe the contents in this repo. I have written the code so far for full precision, and considering only a single GPU, as I did not have a precision sensitive GPU like a V100, or multiple GPUs during my study period.

## Contents

- **A.** [Methods](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Methods.py): A selection of self supervised learning techniques presented as torch.nn.Module objects, which are _mostly_ self-contained apart from files in [Utils](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Utils.py). A special focus is placed on joint-embedding architectures (see SSL cookbook: https://arxiv.org/abs/2304.12210), although the [Masked Autoencoder](https://arxiv.org/abs/2111.06377) is also provided, more of a personal curiosity, but I think the future is joint-embedding based because it is cheaper (you are calculating losses only in the embedding state, not in the dimension of the input that is decoded from the embedding state).

- **B.** [Encoders](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Encoders.py): A selection of image encoder objects, which are not instantiated. You will need to instantiate it using the help provided subsequently in this readme. We have so far in our selection the basic ResNet and Vision Transformer. Instantiation can also be sought from at the bottom end of the code. For ResNet, we have provided all modifications up to [ResNeXT](https://arxiv.org/abs/1611.05431), and for transformers, we have provided the [Relative Attention](https://github.com/microsoft/Swin-Transformer/blob/968e6b5e428186dc99d19166996439c23dccc4d1/models/swin_transformer.py#L77) (link does not contain exact implementation source, but is inspired from it) module so that you may consider it a building block for hybrid architectures.

- **C.** [Schedulers](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Schedulers.py): So far, I only have in this module the one found in [Katsura's github repo](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py). This suits the purpose for now, but **other schedulers are coming soon!**

- **D.** [Utils](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Utils.py): I have here the utilities needed for some of the self-supervised learning techniques, such as the **MultiCropWrapper** and **DinoProjection** head used in [DiNOv1](https://arxiv.org/abs/2104.14294), **SupportSet** for the [NNCLR](https://arxiv.org/abs/2104.14548) method, and **LARS** optimizer copied verbatim from Meta's [DiNO utils](https://github.com/facebookresearch/dino/blob/main/utils.py).

- **E.** [DataAug](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/DataAug.py): Data augmentation techniques, classical (linear transformations to the image), are found here. There are two kinds of data augmentation:
   - **E.1.** Contrastive learning based data augmentation: Outputs a set of 2 (or n) views of transformed input image batch using classical techniques
   - **E.2.** DiNO data augmentation technique. Here, there is a local and a global crop needed for the teacher-student learning approach found in DiNO. Classical augmentation is applied to all crops.
 
There are other elements in the repo, such as [DimensionReduction](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/DimensionReduction.py), which basically is my implementation of the [UMAP](https://arxiv.org/abs/1802.03426) method, from scratch, but it is slow. I recommend using a faster library. Other elements also include the training script [Training](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Training.py), but that only serves as a means to understand how to combine these modules to your data. There is also a [jupyter notebook](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Squeeze_and_attend_mamba.ipynb) which is basically just a little test I was doing on an architecture I call the 'Squeeze and Attend' method combined with a [Mamba](https://arxiv.org/abs/2312.00752) block that I wrote from scratch. Won't go into details, it isn't very useful right now. The Mamba block is slow as it uses the recurrent version of the selective scan, so it won't scale either. I use a standard convolution block prior to the state-space model block in the Mamba block, but that should be replaced with a CUDA optimized causal-conv1d method ([example](https://github.com/Dao-AILab/causal-conv1d)).

## Usage

### Instantiation of a Self-Supervised Learning Method Object

The current repo contains mostly joint-embedding architectures except for one generative method, the Masked Auto-Encoder approach. Each method has been written as a torch.nn.Module PyTorch object, and in order to run, requires a scheduler, dataloader, and optimizer, all of which will be covered in the training subsection at the end of this readme. Most SSL libraries have a separate loss function sent over to the Self-Supervised Learning method, however in this repo, we have the loss function as a feature built within the method. This is perhaps restrictive to you, the user, who would like to experiment with different training loss calculations methods. However, this way we preserve the intent of the authors.

In the following, I will outline the instantiation of several Self-Supervised Learning methods that have been implemented in the Methods file. I have left out the instantiation of the MAE method, as it is not flexible yet, and written only for the ViT architecture.

Instead of being descriptive about the Self-Supervised Learning methods, I will only highlight the paper relevant to each method. Some relevant comments will be there, but they only pertain to what is probably missing in the implementation, or some key things that may help improve performance for you.

- **[SimCLR](https://arxiv.org/abs/2002.05709)**       : Bigger the batch size, better the contrastive learning as infoNCE loss can be better calculated only when the batch size is large, i.e., there are sufficient number of positive and negative examples for the loss estimation.
  ```ruby
  import Methods
  import torch
  device    = torch.device("cuda") # Do not try the 'mps' device as most pytorch functionalities aren't in it.
  model     = Methods.simCLR(
            encoder         = encoder, 
            device          = device, 
            batch_size      = BATCH_SIZE, 
            epochs          = EPOCHS,
            savepath        = './checkpoints/test.pth',
        )
- **[NNCLR](https://arxiv.org/abs/2104.14548)**        : Queue size should be sufficiently large (here the vanilla queue size of 32768 was taken from paper) so that you may estimate nearest neighbors from a big enough population size of samples.
  ```ruby
  import Methods
  import torch
  device    = torch.device("cuda") # Do not try the 'mps' device as most pytorch functionalities aren't in it.
  model     = Methods.NNCLR(
            encoder = encoder,
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
- **[DiNO](https://arxiv.org/abs/2104.14294)**         : The number of global crops is set to 2, where are the number of local crops is passed as a parameter in the instantiation as a variable _CROPS_.
  ```ruby
  import Methods
  import torch
  device    = torch.device("cuda") # Do not try the 'mps' device as most pytorch functionalities aren't in it.
  model     = Methods.DiNO(
            encoder_embedding_dim = encoder_output_dim,       # based only on the global crop size.
            feature_size          = 128,
            encoder               = encoder,
            device                = device,
            batch_size            = BATCH_SIZE,               # Adjust as needed
            epochs                = EPOCHS,                   # Adjust as needed
            temperature_teacher   = 0.04,                     # Example value, adjust as needed
            temperature_student   = 0.07,                     # Example value, adjust as needed
            ncrops                = CROPS,
            savepath              = './checkpoints/test.pth',
        )
  ```
- **[BYOL](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)**         :
  ```ruby
  import Methods
  import torch
  device    = torch.device("cuda") # Do not try the 'mps' device as most pytorch functionalities aren't in it.
  model     = Methods.BYOL(
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
  ```
- **[BarlowTwins](https://arxiv.org/abs/2103.03230)**  :
  ```ruby
  import Methods
  import torch
  device    = torch.device("cuda") # Do not try the 'mps' device as most pytorch functionalities aren't in it.
  model     = Methods.BarlowTwins(
            encoder=encoder,
            device=device,
            batch_size=BATCH_SIZE,
            epochs=50,
            savepath='./drive/MyDrive/SSL_Models/barlowtwins_checkpoint.pth',
            feature_size=FEATURE_SIZE, # important: a large projector output size often improves performance. It might be worthwhile to have this implemented.
            projection_hidden_size_ratio=2,
            gamma=0.0051
        )
  ```
- **[VICReg](https://arxiv.org/abs/2105.04906)**       : Here, the learning rate scheduler per the paper is different from vanilla approaches where the learning rate is the same for all layers. I would recommend looking into it and accordingly implementing your own scheduler to stick to the vanilla VICReg approach.
  ```ruby
  import Methods
  import torch
  device    = torch.device("cuda") # Do not try the 'mps' device as most pytorch functionalities aren't in it.
  model     = Methods.VICReg(
            encoder=encoder,
            device=device,
            epochs=100,
            savepath='./drive/MyDrive/SSL_Models/VICREG_checkpoint.pth',
            batch_size=BATCH_SIZE,
            feature_size=FEATURE_SIZE,
            projection_hidden_size_ratio=2,
            projector_num_layers=2,
            output_projector_size=FEATURE_SIZE*2*2
        )
  ```

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
  
- **Vision Transformer** : The following is how you would instantiate the vision transformer. Provided are instantiation of the vanilla ViT base, tiny and small models. There are, of course, others. These are the ViT Huge and ViT Large, but I never pursued them because my sole focus was on developing self-supervised learning methods first. In this implementation, we have used the pre-norming technique, i.e., layer normalization of the weights are performed prior to the projection operations (if that is correct).

  Lastly, the implementation is image size agnostic. This was necessary for the use of the vision transformer in the DiNO method, as the encoder processes both local and global crops. The approach here is based on Meta's implementation in its DiNO github repo, particularly the [VisionTransformer.interpolate_pos_encoding](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L174).
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
  Mind you, the attention mechanism is quite useless in the low data-regime due to a lack of inductive bias in the attention operation. Even when hybrid architectures are considered, such as the [CoATNnet](https://arxiv.org/abs/2106.04803) paper by Google, you'll see in that paper that the more convolutional blending is preferred in the architecture, the better it generalizes to smaller datasets during training and eval.

  Secondly, I have not implemented a stochastic depth in this vision transformer, so the only regularization is the dropout of certain layers (attention and projection heads). You may take inspiration from the ResNet implementation in my code and proceed with it. In this case, I wonder if stochastic depth should be used in tandem with dropout. I believe people forego one for the other, but you may consider something based on your experience with the methods.

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

### Training on your data
Once the prior instantiations are completed, the approach for training is relatively straightforward. Most methods have a saving of checkpoints every 10 epochs. This is not passed as a variable so you may have to go into the Method object and tweak it yourself.

```ruby
import torch
import Scheduler
dataloader = .... # Take something from prior sections, but be specific to the approach. Besides the DiNO model, all else uses a contrastive style data augmentation object.
SSL_model  = model 
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
```










