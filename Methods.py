import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
from Encoders import PatchEmbed, TransformerBlock
from Scheduler import CustomScheduler
from Utils import SupportSet, MultiCropWrapper, DiNOProjection
import numpy as np
# Method 1: SimCLR


class simCLR(nn.Module): # the similarity loss of simCLR

    def __init__(self, encoder, device,batch_size,epochs,savepath):
        super().__init__()
        self.model = encoder.to(device) # define the encoder here
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.savepath  = savepath
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
        self.scheduler = CustomScheduler(self.optimizer, warmup_epochs=10, initial_lr=1e-4, final_lr=1e-3, total_epochs=epochs)



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

    def get_encoder(self):
        return self.model

    def save_checkpoint(self, file_path):
        """
        Save the model checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path, device):
        """
        Load the model from the checkpoint.
        """
        checkpoint = torch.load(file_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {file_path}")



    def train(self, dataloader):
        self.losses = []  # Track losses

        # Start training
        for epoch in range(self.epochs):
            # Initialize tqdm progress bar
            train_loader = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for views, _ in train_loader:  # Unpack data and labels from each batch
                if views[0].size(0) != self.batch_size:
                    continue  # Skip this batch
                imgs = torch.cat(views, dim=0).to(self.device)
                    
                # Load images and calculate InfoNCE loss
                features = self.model(imgs)
                logits, labels = self.SimCLR_loss(features)
                loss = self.criterion(logits, labels)

                # Append the loss
                self.losses.append(loss.item())

                # Perform optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update tqdm progress bar with the current loss
                train_loader.set_postfix(loss=loss.item())

            if (int(epoch)%10 == 0):
              #file_path = '/content/drive/MyDrive/SimCLR_UMAP/simclr_checkpoint.pth'
              file_path = self.savepath

              # Save the current state of the model and optimizer
              self.save_checkpoint(file_path)


            self.scheduler.step(epoch)


        return self.losses



# Method 2: NNCLR

# NNCLR method



class NNCLR(nn.Module):

    def __init__(self,
                 encoder,
                 feature_size,  # Size of vector at the end of projection operation
                 queue_size,    # Size of the set of nearest neighbor features.
                 projection_hidden_size_ratio, # to be multiplied with encoder size
                 prediction_hidden_size_ratio, # to be multiplied with feature size
                 device,
                 temperature = 0.1, # set sharpness of distribution in cross-entropy loss
                 reduction   = 'mean', # set mean as the reduction in the cross-entropy loss (divides by 1/N, N being batch size)
                 batch_size  = 1000,
                 epochs      = 10,
                 savepath    = '',
                 ):

        super().__init__()

        self.device     = device
        self.batch_size = batch_size
        self.epochs     = epochs
        self.encoder    = encoder.to(device)
        encoder_size    = feature_size
        #self.encoder    = ResNet(ResidualBlock,[3, 5, 7], encoder_size).to(device) # set up the residual block with 1000 feature vector output
        #resnet18 = models.resnet18()
        #resnet18.conv1 = nn.Conv2d(
        #            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        #        )
        #resnet18.maxpool = nn.Identity()
        #self.encoder = nn.Sequential(*list(resnet18.children())[:-1]).to(device)
        #encoder_size = resnet18.fc.in_features

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
        self.savepath           = savepath
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
        #scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-8)#, last_epoch=-1)
        self.scheduler = CustomScheduler(self.optimizer, warmup_epochs=10, initial_lr=1e-4, final_lr=1e-3, total_epochs=epochs)

                     
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
    
    def get_encoder(self):
        return self.encoder

    def save_checkpoint(self, file_path):
        """
        Save the model checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path, device):
        """
        Load the model from the checkpoint.
        """
        checkpoint = torch.load(file_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {file_path}")


    def train(self, dataloader):
        self.losses = []  # Track losses

        # Start training
        for epoch in range(self.epochs):
            # Initialize tqdm progress bar
            train_loader = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for views, _ in train_loader:  # Unpack data and labels from each batch
                if views[0].size(0) != self.batch_size:
                    continue  # Skip this batch
                imgs = torch.cat(views, dim=0).to(self.device)

                # Calculate NNCLR loss
                loss = self.NNCLR_loss(imgs)

                # Append the loss
                self.losses.append(loss.item())

                # Perform optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update tqdm progress bar with the current loss
                train_loader.set_postfix(loss=loss.item())

            # Save model checkpoint at regular intervals
            if epoch % 10 == 0:
                file_path = self.savepath
                # Save the current state of the model and optimizer
                self.save_checkpoint(file_path)

            self.scheduler.step(epoch)

        return self.losses



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






# Method 4) DiNO


class DiNO(nn.Module):

    def __init__(self, encoder_embedding_dim, feature_size, encoder, device, batch_size, epochs,  temperature_teacher, temperature_student, ncrops, savepath, alpha = 0.996):
        super().__init__()
        # both student and teacher are the same encoders
        #self.student    = encoder.to(device)
        #self.teacher    = encoder.to(device)
        # but their heads are different. Here is how using the MultiCropWrapper
        self.student    = MultiCropWrapper(
                            encoder,
                            DiNOProjection(
            in_dim=encoder_embedding_dim,
            out_dim=feature_size,
            use_bn=True,
            norm_last_layer=True,
        )

                          ).to(device)

        self.teacher    = MultiCropWrapper(
                            encoder,
                            DiNOProjection(
            in_dim=encoder_embedding_dim,
            out_dim=feature_size,
            use_bn=True,
            norm_last_layer=False,
        )

                          ).to(device)
        self.savepath  = savepath
        self.device     = device
        self.batch_size = batch_size
        self.epochs     = epochs
        self.optimizer  = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
        self.scheduler  = CustomScheduler(self.optimizer, warmup_epochs=10, initial_lr=1e-6, final_lr=1e-3, total_epochs=self.epochs)
        # DiNO specific parameters
        self.base_momentum  = alpha
        self.final_momentum = 1
        self.alpha      = alpha  # momentum blending constant
        self.t_student  = temperature_student
        self.t_teacher  = temperature_teacher
        self.initial_teacher_temp = temperature_teacher
        self.final_teacher_temp = 0.05 # change this
        self.center_momentum = 0.999 # change this
        self.register_buffer('center', torch.zeros(1, feature_size, device=device))
        self.ncrops     = ncrops

        self.criterion  = torch.nn.CrossEntropyLoss().to(device)

    @torch.no_grad()
    def update_momentum(self, current_epoch):
        # Cosine schedule to update momentum (alpha)
        self.alpha = self.final_teacher_temp - 0.5 * (self.final_teacher_temp - self.final_teacher_temp) * \
                     (1 + np.cos(np.pi * current_epoch / self.epochs))


    @torch.no_grad()
    def update_teacher_temp(self, current_epoch):
        self.t_teacher = self.final_momentum - 0.5 * (self.final_momentum - self.base_momentum) * \
                     (1 + np.cos(np.pi * current_epoch / self.epochs))


    @torch.no_grad()
    def update_teacher(self):
        # This uses the alpha parameter to update all
        # the parameters of the teacher network using
        # that from the student network using momentum.
        #with torch.no_grad():
        for student_params, teacher_params in zip(self.student.parameters(),self.teacher.parameters()):
            teacher_params.data = self.alpha*teacher_params.data + (1.0 - self.alpha)*student_params.data

    @torch.no_grad()
    def update_center(self, teacher_output):
        teacher_output_detached = teacher_output.detach()
        # Update the center with EMA
        self.center = self.center * self.center_momentum + teacher_output_detached.mean(dim=0, keepdim=True) * (1 - self.center_momentum)

    def get_encoder(self):
        return self.teacher, self.student

    def save_checkpoint(self, file_path):

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path, device):

        checkpoint = torch.load(file_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {file_path}")


    def DiNO_loss(self, teacher_output, student_output, current_epoch):

        # Student sharpening
        student_out = student_output / self.t_student
        student_out = student_out.chunk(self.ncrops + 2)

        # teacher centering and sharpening, as well as temperature and center momentum updates
        self.update_teacher_temp(current_epoch)
        teacher_out = F.softmax((teacher_output - self.center) / self.t_teacher, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        self.update_center(teacher_output)


        # Estimation of the loss (ensuring similar views do not contribute to loss)
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss # which is actually the average loss


    def train(self, dataloader):
        self.losses = [] # Track losses

        # Switch off gradient requirement for teacher (I don't know if these steps are redundant)
        for p in self.teacher.parameters():
            p.requires_grad = False
        # Switch off gradient requirement for teacher
        for p in self.student.parameters():
            p.requires_grad = True


        # Start training
        for epoch in range(self.epochs):
            # Initialize tqdm progress bar
            train_loader = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            # Update alpha that adjusts teacher's parameter using momentum over the student's parameters
            self.update_momentum(epoch)

            for batch_idx, (batch_images, _) in enumerate(train_loader):
                if batch_images[0].size(0) != self.batch_size:
                  continue  # Skip this batch


                batch_images = [item.to(self.device) for item in batch_images]
                # Calculate features
                teacher_output = self.teacher(batch_images[:2])
                student_output = self.student(batch_images)

                # Calculate cross-entropy loss
                loss  = self.DiNO_loss(teacher_output,student_output,epoch)

                # Append the loss
                self.losses.append(loss.item())

                # Perform optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update teacher
                self.update_teacher()

                # Update tqdm progress bar with the current loss
                train_loader.set_postfix(loss=loss.item())

            self.scheduler.step(epoch) # update the scheduler learning rate
            # Save model checkpoint at regular intervals
            if epoch % 10 == 0:
                file_path = self.savepath
                # Save the current state of the model and optimizer
                self.save_checkpoint(file_path)

        return self.losses




########### BYOL method




# Define the BYOL method

class BYOL(nn.Module):

    def __init__(self,
                 encoder,
                 device,
                 epochs,
                 savepath,
                 batch_size,
                 feature_size,  # Size of vector at the end of projection operation
                 projection_hidden_size_ratio, # to be multiplied with encoder size
                 prediction_hidden_size_ratio, # to be multiplied with feature size
                 alpha, # momentum parameter
                 ):
        super().__init__()
        # Joint embedding architecture related initialization
        self.student = encoder.to(device)
        self.teacher = encoder.to(device)
        self.student_proj = nn.Sequential(
                            nn.Linear(feature_size,feature_size * projection_hidden_size_ratio),
                            nn.BatchNorm1d(feature_size * projection_hidden_size_ratio),
                            nn.ReLU(),
                            nn.Linear(feature_size * projection_hidden_size_ratio,feature_size * projection_hidden_size_ratio),
                            nn.BatchNorm1d(feature_size * projection_hidden_size_ratio),
                            nn.ReLU(),
                            nn.Linear(feature_size * projection_hidden_size_ratio,feature_size),
                            nn.BatchNorm1d(feature_size)
                            ).to(device)
        self.teacher_proj = self.student_proj.to(device)
        self.student_pred = nn.Sequential(
                            nn.Linear(feature_size,feature_size * prediction_hidden_size_ratio),
                            nn.BatchNorm1d(feature_size * prediction_hidden_size_ratio),
                            nn.ReLU(),
                            nn.Linear(feature_size * prediction_hidden_size_ratio,feature_size)
                            )
        # Inherit the weighted norm from DiNO and add it to the student prediction head
        # Apply weight norm to the last layer of student_pred
        self.student_pred[-1] = self._init_weight_norm_layer(
            feature_size * prediction_hidden_size_ratio,
            feature_size
            )
        # Apply weight initialization to student_pred layers
        self.student_pred.apply(self._init_weights_)
        self.student_pred = self.student_pred.to(device)

        # Training related initialization
        self.device     = device
        self.epochs     = epochs
        self.batch_size = batch_size
        self.optimizer  = torch.optim.AdamW(self.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.05)
        self.scheduler  = Scheduler.CustomScheduler(self.optimizer, warmup_epochs=10, initial_lr=1e-5, final_lr=1e-3, total_epochs=epochs)
        self.savepath   = savepath
        self.alpha      = alpha  # momentum blending constant
        self.final_momentum = 1
        self.base_momentum  = alpha

    def _init_weight_norm_layer(self, in_features, out_features):
        layer = nn.utils.weight_norm(nn.Linear(in_features, out_features, bias=False))
        layer.weight_g.data.fill_(1)  # Initialize weight_g
        layer.weight_g.requires_grad = False  # Disable gradient for weight_g
        return layer

    def _init_weights_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def update_momentum(self,current_epoch):
        self.alpha = self.final_momentum - 0.5 * (self.final_momentum - self.base_momentum) * \
                    (1 + np.cos(np.pi * current_epoch / self.epochs))

    def update_teacher(self):
        with torch.no_grad():
            # update teacher encoder
            for student_params, teacher_params in zip(self.student.parameters(), self.teacher.parameters()):
                teacher_params.data = self.alpha*teacher_params.data + (1.0-self.alpha)*student_params.data
            # update teacher projector
            for student_params, teacher_params in zip(self.student_proj.parameters(), self.teacher_proj.parameters()):
                teacher_params.data = self.alpha*teacher_params.data + (1.0-self.alpha)*student_params.data

    def BYOL_loss(self, x1, x2):
        # Encode
        f1      , f2        = self.teacher(x1), self.student(x2)
        # Project
        proj1   , proj2     = self.teacher_proj(f1), self.student_proj(f2)
        # Student prediction
        pred2               = self.student_pred(proj2)

        proj1 = torch.nn.functional.normalize(proj1, dim=-1, p=2)
        pred2 = torch.nn.functional.normalize(pred2, dim=-1, p=2)
        # Compute the loss

        loss = 2 - 2 * (proj1.detach() * pred2).sum(dim=-1)
        return loss

    def BYOL_loss_symmetric(self, imgs):
        # As discussed in the paper, the BYOL loss is symmetric. This is easy
        # to implement because we just switch the augmented views to recompute
        # the MSE loss. We then add it to the original loss array, and take an
        # average.
        loss        = self.BYOL_loss(imgs[:imgs.shape[0]//2], imgs[imgs.shape[0]//2:])
        loss        += self.BYOL_loss(imgs[imgs.shape[0]//2:], imgs[:imgs.shape[0]//2])

        loss = loss.mean()
        return loss


    def train(self,dataloader):

        self.losses = []

        for epoch in range(self.epochs):
            train_loader = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")


            for views, _ in train_loader:
                if views[0].size(0) != self.batch_size: # skip iter if batch size is smaller than expected
                    continue
                # Concatenate images
                imgs = torch.cat(views, dim=0).to(self.device)

                # Calculate loss
                loss = self.BYOL_loss_symmetric(imgs)
                # Append the loss
                self.losses.append(loss.item())
                # Perform optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Update tqdm progress bar with the current loss and learning rate
                train_loader.set_postfix(loss=loss.item(), lr=current_lr)


            # Update momentum and teacher
            self.update_momentum(epoch)
            self.update_teacher()

            # Save model checkpoint at regular intervals
            if epoch % 10 == 0:
                file_path = self.savepath
                # Save the current state of the model and optimizer
                self.save_checkpoint(file_path)

            self.scheduler.step(epoch)

        return self.losses


    def get_encoder(self):
        return self.teacher, self.student

    def save_checkpoint(self, file_path):

        checkpoint = {
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path, device):

        checkpoint = torch.load(file_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {file_path}")





class VICReg(nn.Module):
    def __init__(self,
                 encoder,
                 device,
                 epochs,
                 savepath,
                 batch_size,
                 feature_size,
                 projection_hidden_size_ratio,
                 projector_num_layers,
                 output_projector_size):
        super().__init__()
        # Joint embedding architecture related initialization
        self.encoder    = encoder.to(device)
        self.projector  = self._create_mlp(feature_size, projection_hidden_size_ratio, output_projector_size, projector_num_layers).to(device)
        
        # Training related initialization
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer  = torch.optim.AdamW(self.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.05)
        #self.scheduler  = Scheduler.CustomScheduler(self.optimizer, warmup_epochs=10, initial_lr=1e-3, final_lr=1e-5, total_epochs=epochs)
        self.scheduler = Scheduler.CosineAnnealingWarmupRestarts(
                        self.optimizer,
                        first_cycle_steps=self.epochs - 10,  # Total epochs minus warm-up epochs
                        cycle_mult=1.0,  # Keep cycle length constant after each restart
                        max_lr=1e-3,  # Maximum LR after warm-up
                        min_lr=1e-5,  # Minimum LR
                        warmup_steps=10,  # Warm-up for 10 epochs
                        gamma=1.0  # Keep max_lr constant after each cycle
                    )
        self.savepath = savepath
        self.alpha    = 25
        self.beta     = 25
        self.gamma    = 1

    def _create_mlp(self, input_size, hidden_size_ratio, output_size, num_layers=3):
        hidden_size = int(input_size * hidden_size_ratio)
        mlp_layers = [nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            mlp_layers += [nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()]
        mlp_layers.append(nn.Linear(hidden_size, output_size))
        return nn.Sequential(*mlp_layers)

    
    def vicreg_loss(self, x1, x2):
        # Encode and Project (Teacher-Student architecture)
        proj1, proj2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))

        # Invariance loss (Mean Squared Error loss)
        invariance_loss = F.mse_loss(proj1, proj2)

        # Variance loss
        std_proj1 = torch.sqrt(proj1.var(dim=0) + 1e-5)
        std_proj2 = torch.sqrt(proj2.var(dim=0) + 1e-5)
        variance_loss = torch.mean(F.relu(1 - std_proj1)) / 2 + torch.mean(F.relu(1 - std_proj2)) / 2

        # Covariance loss
        proj1 = proj1 - proj1.mean(dim=0)
        proj2 = proj2 - proj2.mean(dim=0)
        cov_proj1 = (proj1.T @ proj1) / (proj1.size(0) - 1)
        cov_proj2 = (proj2.T @ proj2) / (proj2.size(0) - 1)
        cov_loss = self.off_diagonal(cov_proj1).pow_(2).sum() / proj1.size(1) + \
                  self.off_diagonal(cov_proj2).pow_(2).sum() / proj2.size(1)

        # Combining all loss components
        loss = (self.alpha*invariance_loss) + (self.beta*variance_loss) + (self.gamma*cov_loss)
        return loss

    def off_diagonal(self, x):
        # Return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def train(self, dataloader):
        self.losses = []

        for epoch in range(self.epochs):
            train_loader = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for views, _ in train_loader:
                if views[0].size(0) != self.batch_size:
                    continue

                imgs = torch.cat(views, dim=0).to(self.device)
                x1, x2 = imgs[:imgs.shape[0]//2], imgs[imgs.shape[0]//2:]
                
                loss = self.vicreg_loss(x1, x2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.losses.append(loss.item())
                train_loader.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])

            
            # Save model checkpoint at regular intervals
            if epoch % 10 == 0:
                file_path = self.savepath
                # Save the current state of the model and optimizer
                self.save_checkpoint(file_path)

            self.scheduler.step(epoch)

        return self.losses

    def get_encoder(self):
        return self.teacher, self.student

    def save_checkpoint(self, file_path):

        checkpoint = {
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path, device):

        checkpoint = torch.load(file_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {file_path}")




