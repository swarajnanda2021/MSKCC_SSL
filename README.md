
# Self-supervised learning library for computer vision
A suite of self-supervised learning techniques and image encoders are provided in this repo, written in PyTorch with ample help from other repos (cited in code), papers, and GPT4.


## Introduction

As preparation for my upcoming postdoctoral venture at MSKCC, or the Memorial Sloan Kettering Cancer Center, I spent a few months studying and implementing various image encoder learning techniques that would leverage the vast amount of image data available at MSKCC upon my arrival. In the following, I will describe the contents in this repo.

### Contents

**A.** [Methods](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Methods.py): A selection of self supervised learning techniques presented as torch.nn.Module objects, which are _mostly_ self-contained apart from files in [Utils](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Utils.py). A special focus is placed on joint-embedding architectures (see SSL cookbook: https://arxiv.org/abs/2304.12210), althought the [Masked Autoencoder](https://arxiv.org/abs/2111.06377) is also provided.

**B.** [Encoders](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Encoders.py): A selection of image encoder objects, which are not instantiated. You will need to instantiate it using the help provided subsequently in this readme. We have so far in our selection the basic ResNet and Vision Transformer. Instantiation can also be sought from at the bottom end of the code. For ResNet, we have provided all modifications up to ResNeXT, and for transformers, we have provided the Relative Attention module so that you may consider it a building block for hybrid architectures.

**C.** [Schedulers](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Schedulers.py):

**D.** [Utils](https://github.com/swarajnanda2021/MSKCC_SSL/blob/main/Utils.py):


