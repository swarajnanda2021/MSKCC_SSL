import torch
import torch.nn as nn


# Encoder 0: Simple ConvNet

class SimpleEncoder(nn.Module):

    # define a init class calling the cifar10 dataloader and only use one example
    def __init__(self, hidden_dim):
        super().__init__() # inherit parent class's attributes
        self.latent_dim = hidden_dim
        self.input_channels = 3

        # define an encoder block for taking two images and feeding it to two encoders
        self.encoder = nn.Sequential(
            # 32 x 32 input
        nn.Conv2d(in_channels=self.input_channels,out_channels=64,kernel_size=4,stride=1,padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2,inplace=False),
        # Output size=[(Input size+2×padding−kernel size) / stride] + 1
        # Conv2d will have out_channels x Output size x Output size
        # which is (32 + 2*1 - 3)/1 + 1 = 32. So 64 x 32 x 32
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2,inplace=False),
        # now it is (32 + 2*1 - 4)/2 + 1 = 16, so 64 x 16 x 16
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2,inplace=False),
        # and now it should be 128 x 8 x 8
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2,inplace=False),
        # and now it should be 256 x 4 x 4
        nn.Conv2d(in_channels=256,out_channels=4*self.latent_dim,kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(4*self.latent_dim),
        nn.LeakyReLU(0.2,inplace=False),
        # and now it should be 512 x 2 x 2
        #nn.Conv2d(in_channels=512,out_channels=4*self.latent_dim,kernel_size=1,stride=1,padding=1),
        #nn.BatchNorm2d(4*self.latent_dim),
        #nn.LeakyReLU(0.2,inplace=False),
        # and now it should be 4*self.latent_dim x 1 x 1
        # This is the vector that is the compressed form of the 3*32*32 image to be fed to the projection
        # head.
        )

        # define a multilayer perceptron block
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*self.latent_dim,self.latent_dim)
        )


    # define a forward function: this should produce the vector representation
    def forward(self,x):
        x = self.encoder(x)
        x = self.MLP(x)
        return x



# Encoder 1: ResNet


class BasicBlock(nn.Module): # ResNet 18 and 34
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Bottleneck Block: ResNet 50, 101 and 152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Convolution Block (Initial Layer)

class ConvBlock(nn.Module):
    def __init__(self,InputChannel,OutputChannel,Kernel,Padding,Stride):
        super().__init__()
        self.input_channel      = InputChannel
        self.kernel             = Kernel
        self.padding            = Padding
        self.stride             = Stride
        self.output_channel     = OutputChannel
        self.convblock          = nn.Sequential(
                nn.Conv2d(
                        in_channels     =self.input_channel,
                        out_channels    =self.output_channel,
                        kernel_size     =self.kernel,
                        stride          =self.stride,
                        padding         =self.padding,
                        bias            =False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=self.kernel,stride=self.padding,padding=self.padding)
        )

    def forward(self,x):
        x = self.convblock(x)
        return x

# ResNet Class
class ResNet(nn.Module):
    def __init__(self, block, layers, outputchannels=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, outputchannels)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
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
        x = x + self.attention(x)
        x = x + self.mlp(x)
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

    def forward(self,x):

        batches             = x.shape[0] # total samples per batch
        x                   = self.patch_embedding(x) # convert images to patch embedding
        class_token         = self.class_token.expand(batches, -1, -1) # 
        x                   = torch.cat((class_token,x), dim=1) # class token is not appended to the patch tokens
        x                   = x + self.position_embedding(x) # Add the position embedding mechanism
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

    def ViTencoder():
        model = ViT_encoder(image_size = 32, 
                    patch_size = 16, 
                    in_channels=3, 
                    embedding_dim = 512, 
                    feature_size= 1000,
                    n_blocks = 12,
                    n_heads = 8,
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
