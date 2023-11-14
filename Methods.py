import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
from Encoders import PatchEmbed, TransformerBlock

# Method 1: SimCLR

class simCLR(nn.Module): # the similarity loss of simCLR

    def __init__(self, encoder, device,batch_size,epochs):
        super().__init__()
        self.model = encoder.to(device) # define the encoder here
        self.optimizer = optim.AdamW(self.model.parameters(),lr=1e-2,weight_decay=1e-2)
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

    def SimCLR_loss(self, features):
        n_views = 2
        # Define the similarity matrix's pattern (which elements are related in the batch and which aren't)
        #original_tensor = torch.arange(0,self.batch_size,1)
        # Create the repeated pattern
        #pattern = torch.repeat_interleave(original_tensor, repeats=n_views)
        pattern = torch.cat([torch.arange(self.batch_size) for i in range(n_views)], dim=0)
        # make similarity matrix by the above method (need to understand this)
        pattern = (pattern.unsqueeze(0) == pattern.unsqueeze(1)).float()
        pattern = pattern.to(self.device)
        mask = torch.eye(pattern.shape[0])
        mask = mask.to(self.device)
        pattern = pattern-mask

        features = F.normalize(features,dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix - mask

        # select and combine positives
        positives = similarity_matrix[pattern.bool()].view(pattern.shape[0],-1)
        negatives = similarity_matrix[~pattern.bool()].view(similarity_matrix.shape[0],-1)

        logits = torch.cat([positives,negatives],dim=1)
        # we have to further develop the logits function as follows: we will define a temperature argument that sets the shape of the distribution
        temperature = 0.07
        logits = logits / temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        return logits, labels

    def train(self, dataloader):

        n_iter = 0
        self.losses = []
        # start training
        for epochs in range(self.epochs):

            train_loader = tqdm(dataloader, desc=f"Epoch {epochs + 1}/{self.epochs}")

            for batch_idx, (views, _) in enumerate(train_loader):    # Stack images from the current batch

                imgs = torch.cat(views, dim=0).to(self.device)


                # load images and calculate infoNCE loss
                #imgs = torch.stack([img for idx in range(batch_size) for img in dataloader[idx][0]], dim=0).to(device)
                features = self.model(imgs)
                logits,labels = self.SimCLR_loss(features)
                loss = self.criterion(logits,labels)

                # Append the loss function
                self.losses.append(loss.item())

                # perform optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                n_iter += 1
        return self.losses



# Method 2: NNCLR

class NNCLR(nn.Module):

    def __init__(self,
                 feature_size,  # Size of vector at the end of projection operation
                 queue_size,    # Size of the set of nearest neighbor features.
                 projection_hidden_size_ratio, # to be multiplied with encoder size
                 prediction_hidden_size_ratio, # to be multiplied with feature size
                 device,
                 temperature = 0.1, # set sharpness of distribution in cross-entropy loss
                 reduction   = 'mean', # set mean as the reduction in the cross-entropy loss (divides by 1/N, N being batch size)
                 batch_size  = 1000,
                 epochs      = 10,
                 ):

        super().__init__()

        self.device     = device
        self.batch_size = batch_size
        self.epochs     = epochs
        #self.encoder    = ResNet(ResidualBlock,[3, 5, 7], encoder_size).to(device) # set up the residual block with 1000 feature vector output
        resnet18 = models.resnet18()
        resnet18.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
        resnet18.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(resnet18.children())[:-1]).to(device)
        encoder_size = resnet18.fc.in_features

        self.projector  = nn.Sequential(
                            nn.Linear(encoder_size,encoder_size * projection_hidden_size_ratio),
                            nn.BatchNorm1d(encoder_size * projection_hidden_size_ratio),
                            nn.ReLU(),
                            nn.Linear(encoder_size * projection_hidden_size_ratio,encoder_size * projection_hidden_size_ratio),
                            nn.BatchNorm1d(encoder_size * projection_hidden_size_ratio),
                            nn.ReLU(),
                            nn.Linear(encoder_size * projection_hidden_size_ratio,feature_size),
                            nn.BatchNorm1d(feature_size)
                            ).to(device)
        self.predictor  = nn.Sequential(
                            nn.Linear(feature_size,feature_size * prediction_hidden_size_ratio),
                            nn.BatchNorm1d(feature_size * prediction_hidden_size_ratio),
                            nn.ReLU(),
                            nn.Linear(feature_size * prediction_hidden_size_ratio,feature_size)
                            ).to(device)

        self.nearest_neighbor   = SupportSet(feature_size=feature_size,queue_size = queue_size).to(device) # This needs to be written next
        self.temperature        = temperature
        self.reduction          = reduction
        self.optimizer          = torch.optim.SGD(self.parameters(), lr=0.3, momentum=0.9, weight_decay=5e-4)
        self.scheduler          = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=0, last_epoch=-1)

    def compute_loss(self,predicted,nn): # supply feature set (passed through projector and predictor) and NNs
        pred_size, _    = predicted.shape
        labels          = torch.arange(pred_size).to(predicted.device)

        nn              = F.normalize(nn,p=2,dim=1) # p is default 2 and dim is default 1
        pred            = F.normalize(predicted,p=2,dim=1)

        logits          = (nn @ pred.T) / self.temperature
        loss            = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss

    def NNCLR_loss(self,imgs):

        x1, x2 = imgs[:imgs.shape[0]//2], imgs[imgs.shape[0]//2:]

        # This bit computes the 'symmetric' loss as proposed by the authors.
        # Mind you the loss is hardly symmetric.

        # Encode the images and pass them through the projector
        f1      , f2       = self.encoder(x1).squeeze()     ,        self.encoder(x2).squeeze()
        proj1   , proj2    = self.projector(f1)             ,        self.projector(f2)
        pred1   , pred2    = self.predictor(proj1)          ,        self.predictor(proj2)
        nn1     , nn2      = self.nearest_neighbor(proj1)   ,        self.nearest_neighbor(proj2)

        # The positives of the images are passed through a predictor, while the images are passed through the SupportSet
        # function to get a batch of Nearest Neighbors from the queue size of 10K (this means the batch size will
        # be much smaller than 10k). However, the authors have decided to make the loss function 'symmetric'.
        # This means that proj1 will pass through a NN function and compared with a prediction of proj2, and
        # vice versa, where proj2 will pass through a NN function and compared with a prediction of proj1.
        # Both these 'branches' will be used to compute the infoNCE loss function.

        # Update Support Set with the projection of features from x1 alone
        self.nearest_neighbor.update(proj1)

        # It is very important to note that Q, the support set, comprises of nearest neighbors
        # of the feature set from x1 alone. So even in the symmetrized loss function, where
        # loss is estimated from feature set of x2, the nearest neighbors are estimated w.r.t. Q,
        # which is based on the projected features of x1.


        # We now move on to computing the loss.

        # It is important to understand the kind of asymmetry that is imposed on the 'symmetric' loss function.
        # The support set (Q) is made only of proj1 of f1.
        # The loss is a mean of infoNCE loss of NNs of f1 in Q, and proj2, and NNs of f2 in Q again.
        nnclr_loss = (self.compute_loss(pred1,nn2) * 0.5) + (self.compute_loss(pred2,nn1) * 0.5)

        return nnclr_loss

    def train(self,dataloader):

        n_iter = 0
        self.losses = []
        # start training
        for epochs in range(self.epochs):

            train_loader = tqdm(dataloader, desc=f"Epoch {epochs + 1}/{self.epochs}")

            for batch_idx, (views, _) in enumerate(train_loader):    # Stack images from the current batch

                image_set = torch.cat(views, dim=0).to(self.device)
                loss = self.NNCLR_loss(image_set)

                # Append the loss 
                self.losses.append(loss.item())

                # perform optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                n_iter += 1
            self.scheduler.step()
        return self.losses

class SupportSet(nn.Module):
    # This object is called to do two things:
    # 1) Store features in a set of a predetermined size, refresh it for every training iteration
    # 2) Query nearest numbers of a feature set from the stored features

    def __init__(self,feature_size,queue_size=10000):
        super().__init__()
        self.feature_size       = feature_size
        self.queue_size         = queue_size
        # make some buffers that aren't considered parameters by pytorch nn.Module
        self.register_buffer("queue", tensor = torch.randn(queue_size,feature_size,dtype = torch.float))
        self.register_buffer("queue_pointer", tensor = torch.zeros(1,dtype=torch.long))

    @torch.no_grad # prevent the following from participating in gradient calculations as it is just a queue update
    def update(self, batch):
        batch_size , _  = batch.shape
        pointer         = int(self.queue_pointer) # it gives an idea of what is the filled state of the queue

        if pointer + batch_size >= self.queue_size:
            # if you've got batches that need to be added to the queue, but the batch exceeds the queue size,
            # then they have to be 'wrapped around' the queue set ideally. But in the implementation
            # I am consulting, the approach simply fills 'up to the brim' and discards the rest of the
            # batch samples. Then it updates the pointer to 0.as_integer_ratio
            self.queue[pointer:,:]                      = batch[:self.queue_size - pointer].detach()
            self.queue_pointer[0]                       = 0

        else:
            self.queue[pointer:pointer + batch_size,:]  = batch.detach()
            self.queue_pointer[0]                       = pointer + batch_size

    def forward(self,x,normalized=True):

        queue_l2    = F.normalize(self.queue,p=2,dim=1)
        x_l2        = F.normalize(x,p=2,dim=1)
        similarity  = x_l2 @ queue_l2.T
        nn_idx      = similarity.argmax(dim=1)

        if normalized:
            out = queue_l2[nn_idx]
        else:
            out = self.queue[nn_idx]

        return out


# Method 3: MAE



class MAE(nn.Module):

    # The initialization will have several commonalities with the ViT, but has a decoder transformer too which sets it apart
    def __init__(self,
                 image_size,            # image_size (int)              : size of the input image
                 patch_size,            # patch_size (int)              : size of the patches to be extracted from the input image
                 in_channels,           # in_channels (int)             : number of input channels
                 embedding_dim,         # embedding_dim (int)           : number of elements of the embedding vector (per patch)
                 feature_size,          # feature_size (int)            : Total size of feature vector
                 n_blocks,              # n_blocks (int)                : total number of sequential transformer blocks (a.k.a. depth)
                 n_heads,               # n_heads (int)                 : total number of attention heads per transformer block
                 mlp_ratio,             # mlp_ratio (float)             : the ratio by which embedding dimension expands inside a transformer block (in the MLP layer after attention)
                 qkv_bias,              # qkv_bias (bool)               : whether to add a bias term to the qkv projection layer or not
                 attention_dropout,     # attention_dropout (float)     : dropout in the attention layer
                 projection_dropout,    # projection_dropout (float)    : dropout in the projection layer
                 mask_ratio,            # mask_ratio (float)            : masking applied to input image patch embeddings
                 decoder_embedding_dim, # decoder embedding dim (float) : decoder has a different embedding dim for convenience
                 decoder_n_heads,
                 decoder_n_blocks,
                 epochs,
                 device,
                 batch_size
                 ):
        super().__init__()

        # Part 1) Important elements for the MAE method alone
        self.mask_ratio         = mask_ratio
        self.patch_embedding    = PatchEmbed(
                                            img_size        =   image_size,
                                            patch_size      =   patch_size,
                                            in_channels     =   in_channels,
                                            embed_dim       =   embedding_dim
                                            )

        self.class_token        = nn.Parameter(torch.zeros(1,1,embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patches + 1, embedding_dim), requires_grad=False)
        # In the above line, the requires_grad = False appeared to fix the sin-cos embedding (whatever that means)

        # Part 2) Encoder specific elements, this should ideally be replaced by a self.encoder = encoder function
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

        # Part 3) Decoder specific elements
        self.decoder_embedding      = nn.Linear(embedding_dim, decoder_embedding_dim, bias=True)
        self.mask_token             = nn.Parameter(torch.zeros(1,1,decoder_embedding_dim))
        self.decoder_pos_embedding  = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patches + 1, decoder_embedding_dim), requires_grad=False)
        self.dec_blocks             = nn.ModuleList(
                                        [
                                            TransformerBlock(
                                                            embedding_dim       = decoder_embedding_dim,
                                                            num_heads           = decoder_n_heads,
                                                            MLP_ratio           = mlp_ratio,
                                                            qkv_bias            = qkv_bias,
                                                            attention_dropout   = attention_dropout,
                                                            projection_dropout  = projection_dropout,
                                                             )
                                         for _ in range(decoder_n_blocks)]
                                        )
        self.dec_norm               = nn.LayerNorm(decoder_embedding_dim, eps = 1e-6)
        self.dec_head               = nn.Linear(decoder_embedding_dim, patch_size**2 * in_channels, bias=True)

        # Optimization related stuff:

        #self.optimizer          = torch.optim.SGD(self.parameters(), lr=0.06, momentum=0.9, weight_decay=5e-4)
        #
        self.optimizer          = torch.optim.AdamW(self.parameters(), lr=(1.5e-4) * batch_size / 256, betas=(0.9, 0.95), weight_decay=0.05)
        self.scheduler          = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-8)#, last_epoch=-1)

        self.epochs             = epochs
        self.device             = device

        # Meta has a normalized pixel loss implemented, but we will avoid it for the time being

    def RandomMasking(self, patches):

        N, L, D         = patches.shape # batch, length, dimension
        retained        = int((1-self.mask_ratio)*L) # The total number of patches that will be retained

        noise_vec       = torch.rand(N, L, device = patches.device) # Generate a noise matrix using torch.rand function b/w [0 1]
        id_shuffled     = torch.argsort(noise_vec,dim=1) # this provides indices to shuffle, lower ones are kept, higher ones are removed, per self.ratio
        id_restore      = torch.argsort(id_shuffled,dim = 1) # this undoes the shuffling (for posterity)

        id_keep         = id_shuffled[:, :retained]        # use the first 'retained' elements and discard the rest
        x_masked        = torch.gather(patches, dim = 1, index = id_keep.unsqueeze(-1).repeat(1,1,D))

        # The above line will achieve the following:
        # given x = [1 2 0 ; 0 1 2], for example,
        # you get: x_masked = [ [1 ; 2 ; 0] ; [0 ; 1 ; 2] ],
        # the repeat will enable the entries ([1 ; 2 ; 0] and [0 ; 1 ; 2]) to be repeated D times along the column

        mask                = torch.ones( [N , L] , device = patches.device)
        mask[:, :retained]  = 0

        mask                = torch.gather(mask, dim = 1, index = id_restore)

        return x_masked, mask, id_restore

    def encode(self, img_set , mask_ratio):
        # x has dim: B, C, H, W
        x = self.patch_embedding(img_set)           # Convert image to patch embedding:-> Batch, N_patches, Embedding_dim
        x = x + self.position_embedding[:, 1:, :]   # Position embedding avoids the first entry in first dim as that is the location of the class token
        # x has dim: B, N_patches, Embedding_dim

        x, mask, ids_restore = self.RandomMasking(x)

        # Now add the class token, basically complete the vanilla bits of the ViT
        cls_token   = self.class_token + self.position_embedding[:,:1,:]
        cls_tokens  = cls_token.expand(x.shape[0],-1,-1)
        x           = torch.cat((cls_tokens,x),dim=1)
        # Understand what has happened above by understanding matrix sizes

        # Add the transformer blocks, layer norm
        for block in self.blocks:
            x       = block(x)
        x           = self.norm(x)

        return x, mask, ids_restore

    def decode(self, embeddings, ids_restore):
        # first convert embedding dim to decoder specific embedding dim (can both be same too, I suppose)
        x           = self.decoder_embedding(embeddings)

        # append mask token
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_          = torch.cat([x[:,1:,:], mask_tokens], dim=1)
        x_          = torch.gather(x_, dim=1, index= ids_restore.unsqueeze(-1).repeat(1,1,x.shape[2]))
        x           = torch.cat([x[:,:1,:], x_], dim=1) # appending the class token
        # Understand what has happened above by understanding matrix sizes

        x           = x + self.decoder_pos_embedding    # add pos embed

        # Apply transformer module and the layer norm
        for block in self.dec_blocks:
            x       = block(x)
        x           = self.dec_norm(x)

        # Apply the prediction head to convert embeddings to patch^2 * n_channels
        x           = self.dec_head(x)

        # Remove class token
        x           = x[:,1:,:]

        return x

    def patchify(self, image_set):
        # converts images to patches so that we can calculate the mean squared error for loss estimation
        # image set : B, C, H, W
        # x         : B, L, patch**2 *3

        p           =   self.patch_embedding.patch_size
        h = w       =   image_set.shape[2] // p
        x           =   image_set.reshape(shape = (image_set.shape[0] , 3 , h , p , w , p))
        x           =   torch.einsum('nchpwq->nhwpqc', x)
        x           =   x.reshape(shape=(image_set.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        # bring patches back to image dimension so that you can plot the image(s) generated
        # x         : B, L, patch**2 *3
        # image set : B, C, H, W

        p           = self.patch_embedding.patch_size
        h = w       = int(x.shape[1] ** 0.5)
        x           = x.reshape(shape = (x.shape[0], h, w, p, p, 3))
        x           = torch.einsum('nhwpqc->nchpwq', x)
        image_set   = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return image_set

    def MAE_loss(self, image_set, pred, mask):
        # This computes the loss between a given image set and the predictions (incl. mask)
        #   imgs: [N, 3, H, W]
        #   pred: [N, L, p*p*3]
        #   mask: [N, L], 0 is keep, 1 is remove,

        target = self.patchify(image_set)
        loss = (target - pred) ** 2
        loss = loss.mean(dim=-1)
        mask_sum = mask.sum() + 1e-6  # Adding a small epsilon to avoid division by zero
        loss = (loss * mask).sum() / mask_sum
        return loss


    def train(self, dataloader):

        self.losses = []
        # start training
        for epoch in range(self.epochs):

            # Initialize tqdm
            train_loader = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for batch_idx, (images, _) in enumerate(train_loader):    # Stack images from the current batch

                images = images.to(self.device)
                x, mask, ids_restore = self.encode(images, self.mask_ratio)
                pred = self.decode(x, ids_restore)

                loss = self.MAE_loss(images, pred, mask)

                # perform optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Append the loss
                self.losses.append(loss.item())

                # Update tqdm postfix to show the current batch loss
                train_loader.set_postfix(batch_loss=loss.item())

            self.scheduler.step()

        return self.losses


