import torch
import torch.nn as nn
import torch.nn.functional as F



# The following class creates a SupportSet object which stores representations made during a contrastive training process for methods like NNCLR
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

