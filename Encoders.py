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


class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride = 1, downsample = None):
        super().__init__() # inherit properties of nn.Module
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3,stride=stride,padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels,out_channels=out_channels,stride=1,padding=1,kernel_size=3),
                                   nn.BatchNorm2d(out_channels)
                                   )
        self.out_channels = out_channels
        self.nonlinear = nn.ReLU()
        self.downsample = downsample

    def forward(self,x):
        residual = x #save residual in separate variable
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual # add the residual connections
        out = self.nonlinear(out)
        return out


class ResNet(nn.Module):
    def __init__(self,block,blocks,outputchannels):
        super().__init__()
        # add initial convolutional layer
        self.convlayer  = ConvBlock(InputChannel=3,OutputChannel=64,Kernel=3,Padding=1,Stride=1)
        # add the residual blocks
        self.layer1     = self._make_layer(block,inchannels=64,channels=128,numblocks = blocks[0],stride=2)
        self.layer2     = self._make_layer(block,inchannels=128,channels=256,numblocks = blocks[1],stride=2)
        self.layer3     = self._make_layer(block,inchannels=256,channels=512,numblocks = blocks[2],stride=2)
        #self.layer4     = self._make_layer(block,inchannels=64,channels=128,layers[0],stride=1)
        # add the average pooling block
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1)) # compresses the above to 512,1,1 output size by averaging over the other dimensions
        self.fc         = nn.Linear(512,outputchannels)

    def _make_layer(self,block,inchannels,channels,numblocks,stride=1):
        # first define whether a downsample is needed:
        downsample = None
        if stride != 1 or inchannels != channels:
            downsample  = nn.Sequential(
                nn.Conv2d(in_channels=inchannels,out_channels = channels, kernel_size=1,stride = stride),
                nn.BatchNorm2d(channels)
            )
        layers = []
        layers.append(block(inchannels,channels,stride,downsample))
        for _ in range(1,numblocks): # loop over number of blocks
            layers.append(block(channels,channels))

        return nn.Sequential(*layers) # * operator is used to unpack the elements of an iterable (layers)

    def forward(self,x):
        x = self.convlayer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpooling(x)
        x = torch.flatten(x,1) # convert to 1X1 vector
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
        self.dropout        = dropout
        self.activation     = nn.GELU

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
        for block in range(self.blocks):
            x = block(x)
        x                   = self.norm(x) # add the layer norm mechanism now, giving us n_samples X (class token + patch token) X embedding dim
        x                   = x[:, 1:, :].mean(dim=1)  # global pool without cls token, giving us n_samples X embedding_dim 
        # the 1: is done in the second dim because the first entry there is the class token, which we do not need (why do we have it then? lol...)
        x                   = self.head(x) # expand feature set to intended feature size
        return x

        
if __name__ == "__main__":
    
    def resnet34():
        layers=[3, 5, 7, 5]
        model = ResNet(ResidualBlock, layers,1000)
        return model
    
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