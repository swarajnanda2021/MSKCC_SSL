
# Self-supervised learning library for computer vision
A suite of self-supervised learning techniques and image encoders are provided in this repo, written in PyTorch with ample help from other repos (cited in code), papers, and GPT4.


## Introduction

As preparation for my upcoming postdoctoral venture at MSKCC, or the Memorial Sloan Kettering Cancer Center, I spent a few months studying and implementing various image encoder learning techniques that would leverage the vast amount of image data available at MSKCC upon my arrival. In the following, I will describe the contents in this repo. I have written the code so far for full precision, and considering only a single GPU, as I did not have a precision sensitive GPU like a V100, or multiple GPUs during my study period.

### Contents

- **A.** [Methods](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Methods.py): A selection of self supervised learning techniques presented as torch.nn.Module objects, which are _mostly_ self-contained apart from files in [Utils](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Utils.py). A special focus is placed on joint-embedding architectures (see SSL cookbook: https://arxiv.org/abs/2304.12210), althought the [Masked Autoencoder](https://arxiv.org/abs/2111.06377) is also provided.

- **B.** [Encoders](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Encoders.py): A selection of image encoder objects, which are not instantiated. You will need to instantiate it using the help provided subsequently in this readme. We have so far in our selection the basic ResNet and Vision Transformer. Instantiation can also be sought from at the bottom end of the code. For ResNet, we have provided all modifications up to ResNeXT, and for transformers, we have provided the Relative Attention module so that you may consider it a building block for hybrid architectures.

- **C.** [Schedulers](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Schedulers.py): So far, I only have in this module the one found in [Katsura's github repo](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py). This suits the purpose for now, but **other schedulers are coming soon!**

- **D.** [Utils](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Utils.py): I have here the utilities needed for some of the self-supervised learning techniques, such as the **MultiCropWrapper** and **DinoProjection** head used in [DiNOv1](https://arxiv.org/abs/2104.14294), **SupportSet** for the [NNCLR](https://arxiv.org/abs/2104.14548) method, and **LARS** optimizer copied verbatim from Meta's [DiNO utils](https://github.com/facebookresearch/dino/blob/main/utils.py).

- **E.** [DataAug](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/DataAug.py): Data augmentation techniques, classical (linear transformations to the image), are found here. There are two kinds of data augmentation:
   - **E.1.** Contrastive learning based data augmentation: Outputs a set of 2 (or n) views of transformed input image batch using classical techniques
   - **E.2.** DiNO data augmentation technique. Here, there is a local and a global crop needed for the teacher-student learning approach found in DiNO. Classical augmentation is applied to all crops.


