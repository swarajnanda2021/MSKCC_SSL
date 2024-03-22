import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Utils import PreNorm

# Encoder 1: ResNet with many modifications (incl. resnext)
class DownsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, modification_type={''}):
        super(DownsampleModule, self).__init__()
        if 'resnetD' in modification_type: # add average pooling if resnetD is present
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else: #just use the regular stuff # 'standard' in modification_type or 'resnetB' in modification_type:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        

    def forward(self, x):
        return self.downsample(x)


class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels,reduction = 0.25): # keep the reduction fixed 
        super(SqueezeAndExcite, self).__init__()
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels * reduction), in_channels),
            nn.Sigmoid()                      
            )
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)

        return x * y.expand_as(x)


class BasicBlock(nn.Module): # ResNet 18 and 34
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, modification_type={''}, dropoutprob = 0.0):
        super(BasicBlock, self).__init__()

        self.prob_dropout = dropoutprob

        conv1_stride = 1
        conv2_stride = stride
        if 'resnetB' in modification_type and stride != 1:
            conv1_stride = 1
            conv2_stride = 2
        if 'resnext' in modification_type: 
            # if you want resnext we'll replace the 3x3 conv with a grouped conv with 32 groups
            groups = 32
        else:
            groups = 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=conv1_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if 'squeezeandexcite' in modification_type and stride != 1:
            self.squeezeandexcite1 = SqueezeAndExcite(out_channels)
        else:
            self.squeezeandexcite1 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride = conv2_stride,padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if 'squeezeandexcite' in modification_type and stride != 1:
            self.squeezeandexcite2 = SqueezeAndExcite(out_channels)
        else:
            self.squeezeandexcite2 = nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.squeezeandexcite1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.squeezeandexcite2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.training and self.prob_dropout > 0.0 and self.downsample is None:
          
          # Calculate the survival probability of this block. Make a binary choice between 0 and 1 based on self.prob_dropout
          survival_prob = 1.0 - self.prob_dropout
          random_tensor = torch.rand([], dtype=out.dtype, device=out.device) + survival_prob
          binary_tensor = random_tensor.floor()  # This will be 1.0 with probability 'survival_prob'
          out = out * binary_tensor
        
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, modification_type={''}, dropoutprob = 0.0):
        super(Bottleneck, self).__init__()

        self.prob_dropout = dropoutprob

        conv1_stride = 1
        conv2_stride = stride
        if 'resnetB' in modification_type and stride != 1:
            conv1_stride = 1
            conv2_stride = 2

        if 'resnext' in modification_type: 
            # if you want resnext we'll replace the 3x3 conv with a grouped conv with 32 groups
            groups = 32
        else:
            groups = 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=conv1_stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=conv2_stride, padding=1,bias=False, groups = groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if 'squeezeandexcite' in modification_type and stride != 1: # all 3x3 convs after stage 1 have squeeze and excite function
            self.squeezeandexcite2 = SqueezeAndExcite(out_channels)
        else:
            self.squeezeandexcite2 = nn.Identity()

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        if 'preactivation_residual_unit' in modification_type:
            self.preact_residual = True
        else
            self.preact_residual = False
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        # return relu of x if survival is not possible
        if self.training and self.prob_dropout > 0.0 and self.downsample is None:
          
          # Calculate the survival probability of this block. Make a binary choice between 0 and 1 based on self.prob_dropout
          survival_prob = 1.0 - self.prob_dropout
          random_tensor = torch.rand([], dtype=x.dtype, device=x.device) + survival_prob
          binary_tensor = random_tensor.floor()  # This will be 1.0 with probability 'survival_prob'
          if binary_tensor.item() == 0.0:
            return F.relu(identity, inplace=False)

        if self.preact_residual:
            x = self.relu(x)
        # else just continue with block processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.squeezeandexcite2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        if not self.preact_residual:
            x = self.relu(x) 

        return x



class MBConv(nn.Module):
    def __init__(self, inp, oup, expansion, downsample):
        super().__init__()
        self.downsample = downsample
        stride = 1 if not downsample else 2
        hidden_dim = int(expansion * inp)

        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                          padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, 
                          padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SqueezeAndExcite(hidden_dim, expansion=0.25),
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup)
            )

        self.conv = PreNorm(norm=nn.BatchNorm2d, model=self.conv, dimension=inp)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return self.proj(x) + self.conv(x)



class ResNet(nn.Module):
    def __init__(self, block, layers, outputchannels=1000, modification_type={''}):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Input branch
        if 'resnetC' in modification_type:
          self.input_branch = nn.Sequential(
              nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
              nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
          
              nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
              nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
          
              nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),

          )
          
        else:
          self.input_branch = nn.Sequential(
              nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              
          )
        
        # Maxpooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual branch stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, modification_type = {''}) # no tweaks whatsoever to the first except if resnext is chosen
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, modification_type=modification_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, modification_type=modification_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, modification_type=modification_type)
        
        # Final average pool and fully connected linear layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, outputchannels)

    def _make_layer(self, block, out_channels, blocks, stride=1, modification_type={''}):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion: # only for the first layer
            downsample = DownsampleModule(self.in_channels, out_channels * block.expansion, stride, modification_type)

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, modification_type, dropoutprob = 0.0))
        self.in_channels = out_channels * block.expansion
        for block_idx in range(1, blocks):
          if 'stochastic_depth' in modification_type:
            dropoutprob = 0.5 * (block_idx / (blocks - 1))  
          else:
            dropoutprob = 0.0

          layers.append(block(self.in_channels, out_channels, dropoutprob = dropoutprob))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        x = self.input_branch(x)
        
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Encoder 2: ViT

class PatchEmbed(nn.Module):
    # converts image into patch embeddings based on total number of non-overlapping crops.
    # For each image containing n patches, there should be n embedding vectors per image, so a n x embedding_vector matrix.    
    def __init__(self,img_size,patch_size,in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size       = img_size
        self.patch_size     = patch_size
        self.in_channels    = in_channels
        self.n_patches      = (img_size // patch_size)**2
        self.project        = nn.Conv2d(
                                    in_channels     =in_channels,
                                    out_channels    = embed_dim,
                                    kernel_size     = patch_size,
                                    stride          = patch_size,
                                    )
    
    def forward(self,x):
        # x has input a tensor of shape B, C, H, W (batch, channel, height, width)

        x = self.project(x)     # Batch X Embedding Dim X sqrt(N_patches) X sqrt(N_patches)
        x = x.flatten(2)        # Batch X Embedding Dim X N_patches
        x = x.transpose(1,2)    # Batch X N_patches X Embedding Dim

        return x


class Attention(nn.Module):

    def __init__(self, embed_dim, n_heads, qkv_bias = False, attn_dropout = 0., projection_dropout=0.):
        super().__init__()
        self.embed_dim          = embed_dim
        self.n_heads            = n_heads
        self.head_dim           = embed_dim // n_heads
        self.scale              = self.head_dim ** -0.5 # From vaswani paper
        self.qkv                = nn.Linear(embed_dim, 3* embed_dim) # convert input to query, key and value
        self.project            = nn.Linear(embed_dim,embed_dim)
        self.project_dropout    = nn.Dropout(projection_dropout)
        self.attention_dropout  = nn.Dropout(attn_dropout)

    def forward(self,x):

        batches, tokens, embed_dim = x.shape # tokens = total patches plus 1 class token 

        QueryKeyValue = self.qkv(x) # it is like a neural form of repmat function.
        QueryKeyValue = QueryKeyValue.reshape(batches, tokens, 3, self.n_heads,self.head_dim) 
        # Above has following dim: batches, tokens, [Query  Key Value], num_heads, head_dim
        QueryKeyValue = QueryKeyValue.permute(      2,      0, 3,             1,           4) 
        # Above has following dim: QKV, batches, num_heads, tokens, head_dim
        Query, Key, Value    = QueryKeyValue[0], QueryKeyValue[1], QueryKeyValue[2]
        # Above has following dim: batches, num_heads, tokens, head_dim
        Attn_dot_product     = (Query @ Key.transpose(-2, -1)) * self.scale
        # Above has following dim: batches, num_heads, tokens, tokens
        Attention_mechanism  = Attn_dot_product.softmax(dim=-1)
        # Above has following dim: batches, num_heads, tokens, tokens
        Attention_mechanism  = self.attention_dropout(Attention_mechanism)
        # Applying the mask (from Values)
        Masking_mechanism    = (Attention_mechanism @ Value).transpose(1,2)
        # Above has following dim: batches, tokens, num_heads, head_dimension
        Masking_mechanism    = Masking_mechanism.flatten(2)
        # Above has following dim: batches, tokens, (num_heads*head_dimension), or, batches, tokens, embedding_dim
        Projection_operation = self.project(Masking_mechanism)
        Projection_operation = self.project_dropout(Projection_operation)

        return Projection_operation 


class RelativeAttention(nn.Module): # Based on Vasvani's 2018 paper https://arxiv.org/pdf/1803.02155.pdf and implementations from Swin transformer official codebase
    def __init__(self, inp, oup, image_size, patch_size, heads=8, projection_dropout=0., attn_dropout = 0.):
        super().__init__()
        

        self.embed_dim          = inp
        self.n_heads            = heads
        self.head_dim           = self.embed_dim // self.n_heads
        self.scale              = self.head_dim ** -0.5 # From vaswani paper
        self.qkv                = nn.Linear(self.embed_dim, 3* self.embed_dim) # convert input to query, key and value
        self.project            = nn.Linear(self.embed_dim,oup)
        self.projection_dropout = nn.Dropout(projection_dropout)
        self.attention_dropout  = nn.Dropout(attn_dropout)


        # parameter table of relative position bias 
        # (comes from the window attention module in swin transformers)
        self.ih, self.iw   = image_size 
        self.ih = int(self.ih/patch_size)
        self.iw = int(self.iw/patch_size)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads)) 

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1,2,0).contiguous()

        # The following lines implement the skewing of the relative_coords matrix.
        # This eliminates the need for the query and relative coordinate dor product
        # thus reducing the computational cost of the process. From music
        # transformers paper: https://arxiv.org/pdf/1809.04281.pdf (same Vaswani guyl, dude's a legend)
        relative_coords[:,:,0] += self.ih - 1
        relative_coords[:,:,1] += self.iw - 1 # these lines make relative coordinates position positive, as prior subtraction makes them negative
        relative_coords[:,:,0] *= 2 * self.iw - 1 # this scales the distance of the pixel (patch) positions in a new row of the image so that it is more than the last pixel (patch) in the previous row
        relative_index = relative_coords.sum(-1) # will be num patches^2 X num patches^2
        self.register_buffer("relative_index", relative_index) 

        
        
    def forward(self, x):

        batches, tokens, embed_dim = x.shape # tokens = total patches plus 1 class token

        QueryKeyValue = self.qkv(x) # it is like a neural form of repmat function.
        QueryKeyValue = QueryKeyValue.reshape(batches, tokens, 3, self.n_heads,self.head_dim)
        # Above has following dim: batches, tokens, [Query  Key Value], num_heads, head_dim
        QueryKeyValue = QueryKeyValue.permute(      2,      0, 3,             1,           4)
        # Above has following dim: QKV, batches, num_heads, tokens, head_dim
        Query, Key, Value    = QueryKeyValue[0], QueryKeyValue[1], QueryKeyValue[2]
        # Above has following dim: batches, num_heads, tokens, head_dim
        Attn_dot_product     = (Query @ Key.transpose(-2, -1)) * self.scale
        
        
        #Estimate the relative position bias and add it to the attention operation
        relative_position_bias = self.relative_position_bias_table[self.relative_index.view(-1)].view(
            self.ih * self.iw,  self.ih * self.iw, -1
        )
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous()
        
        
        
        Attn_dot_product +=  relative_position_bias.unsqueeze(0)
        # Above has following dim: batches, num_heads, tokens, tokens
        Attention_mechanism  = Attn_dot_product.softmax(dim=-1)
        Attention_mechanism  = self.attention_dropout(Attention_mechanism)
        
        Attention_mechanism  = Attn_dot_product.softmax(dim=-1)
        # Above has following dim: batches, num_heads, tokens, tokens
        
        
        # Applying the mask (from Values)
        Masking_mechanism    = (Attention_mechanism @ Value).transpose(1,2)
        
        # Above has following dim: batches, tokens, num_heads, head_dimension
        Masking_mechanism    = Masking_mechanism.flatten(2)
        # Above has following dim: batches, tokens, (num_heads*head_dimension), or, batches, tokens, embedding_dim
        Projection_operation = self.project(Masking_mechanism)
        Projection_operation = self.projection_dropout(Projection_operation)



        return Projection_operation




class MultiLayerPerceptron(nn.Module):

    def __init__(self,in_features,hidden_features,out_features,dropout=0.):
        super().__init__()
        self.fc1            = nn.Linear(in_features,hidden_features)
        self.fc2            = nn.Linear(hidden_features,out_features)
        self.dropout        = nn.Dropout(dropout)
        self.activation     = nn.GELU()

    def forward(self,x): # x :: batches, tokens, in features
        x = self.fc1(x) # x :: batches, tokens, hidden features
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x) # x :: batches, tokens, out features
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, MLP_ratio=4.0, qkv_bias = True, attention_dropout=0., projection_dropout=0.):
        super().__init__()
        self.norm1      = nn.LayerNorm(embedding_dim,eps=1e-6)
        self.norm2      = nn.LayerNorm(embedding_dim,eps=1e-6)
        self.attention  = Attention(embedding_dim,num_heads,qkv_bias,attention_dropout,projection_dropout)
        hidden_features = int(MLP_ratio * embedding_dim)
        self.mlp        = MultiLayerPerceptron(embedding_dim, hidden_features, embedding_dim, projection_dropout)

    def forward(self,x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT_encoder(nn.Module):
    
    def __init__(self, 
                 image_size,            # image_size (int)            : size of the input image
                 patch_size,            # patch_size (int)            : size of the patches to be extracted from the input image
                 in_channels,           # in_channels (int)           : number of input channels
                 embedding_dim,         # embedding_dim (int)         : number of elements of the embedding vector (per patch)  
                 feature_size,          # feature_size (int)          : Total size of feature vector
                 n_blocks,              # n_blocks (int)              : total number of sequential transformer blocks (a.k.a. depth)
                 n_heads,               # n_heads (int)               : total number of attention heads per transformer block
                 mlp_ratio,             # mlp_ratio (float)           : the ratio by which embedding dimension expands inside a transformer block (in the MLP layer after attention)
                 qkv_bias,              # qkv_bias (bool)             : whether to add a bias term to the qkv projection layer or not
                 attention_dropout,     # attention_dropout (float)   : dropout in the attention layer
                 projection_dropout     # projection_dropout (float)  : dropout in the projection layer
                 ):
        super().__init__()
        self.patch_embedding    = PatchEmbed(
                                            img_size        =   image_size,
                                            patch_size      =   patch_size,
                                            in_channels     =   in_channels,
                                            embed_dim       =   embedding_dim
                                            )
        
        self.class_token        = nn.Parameter(torch.zeros(1,1,embedding_dim)) 
        self.position_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patches + 1, embedding_dim))
        self.position_dropout   = nn.Dropout(p = projection_dropout)
        self.blocks             = nn.ModuleList(
                                        [
                                            TransformerBlock(
                                                            embedding_dim       = embedding_dim,
                                                            num_heads           = n_heads,
                                                            MLP_ratio           = mlp_ratio,
                                                            qkv_bias            = qkv_bias,
                                                            attention_dropout   = attention_dropout,
                                                            projection_dropout  = projection_dropout,
                                                             )
                                         for _ in range(n_blocks)]
                                        )
        self.norm               = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.head               = nn.Linear(embedding_dim, feature_size)

    def pos_embedding_interp(self, x, h, w):

        num_patches = x.shape[1] - 1 # because one is a class token
        N = self.position_embedding.shape[1] - 1 # this is the shape the ViT expects

        if num_patches == N: # won't include a check for the image being square
          return self.position_embedding.shape[1] # because no interpolation needs to be done
        # Now we need to do interpolation. So begin by separating class and position tokens
        class_pos_embed   = self.position_embedding[:,0]
        patch_pos_embed   = self.position_embedding[:,1:]
        dim         = x.shape[-1] # patch embedding dimensionality
        w0 = w // self.patch_embedding.patch_size
        h0 = h // self.patch_embedding.patch_size
        w0, h0 = w0+0.1, h0+0.1 # preventing some division by zero (just in case)

        # Perform the interpolation
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    
    def forward(self,x):

        batches, _, W, H = x.shape # B, C, W, H

        x                   = self.patch_embedding(x) # convert images to patch embedding
        class_token         = self.class_token.expand(batches, -1, -1) #
        x                   = torch.cat((class_token,x), dim=1) # class token is not appended to the patch tokens

        # In classical vision transformers, the position embedding is strictly
        # unchanged in the forward pass, this is because the ViT expects only one
        # image size, none else.
        # x                   = x + self.position_embedding # Add the position embedding mechanism
        # However, we want a variable positional encoding, allowing us to
        # use the same ViT architecture for multiple image sizes.
        x                   = x + self.pos_embedding_interp(x, H, W)
        x                   = self.position_dropout(x)
        for block in self.blocks:
            x = block(x)
        x                   = self.norm(x) # add the layer norm mechanism now, giving us n_samples X (class token + patch token) X embedding dim
        x                   = x[:, 1:, :].mean(dim=1)  # global pool without cls token, giving us n_samples X embedding_dim
        # the 1: is done in the second dim because the first entry there is the class token, which we do not need (why do we have it then? lol...)
        x                   = self.head(x) # expand feature set to intended feature size
        return x

        
if __name__ == "__main__":
    
    def resnet18():# 11.439168M parameters
        layers=[2, 2, 2, 2]
        model = ResNet(block = BasicBlock, layers = layers,outputchannels = 512)
        return model
        
    def resnet34():# 21.547328M parameters
        layers=[3, 4, 6, 3]
        model = ResNet(block = BasicBlock, layers = layers,outputchannels = 512)
        return model
    
    def resnet50():# 24.557120M parameters
        layers=[3, 4, 6, 3]
        return ResNet(block = Bottleneck, layers = layers, outputchannels = 512)

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
