import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import models

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

