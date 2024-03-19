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

    @torch.no_grad() # prevent the following from participating in gradient calculations as it is just a queue update
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





# We write now the DiNO projection heads, and a wrapper around the dino backbone and projection
# head which allows it to take in images of various different sizes.


class DiNOProjection(nn.Module):

    def __init__(
        self,
        in_dim      ,
        out_dim     ,
        hidden_dim        =   512,
        bottleneck_dim    =   256,
        use_bn            =   False,
        norm_last_layer   =   True,
        nlayers           =   3,
    ):
      super().__init__()

      nlayer = max(nlayers,1) # set the total number of MLP layers
      if nlayers == 1:

        self.mlp = nn.Linear(in_dim, bottleneck_dim)

      else:

        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn == True:
          layers.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(nlayers-2):
          layers.append(nn.Linear(hidden_dim,hidden_dim))
          if use_bn == True:
            layers.append(nn.BatchNorm1d(hidden_dim))
          layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim,bottleneck_dim))

        self.mlp = nn.Sequential(*layers)

      self.apply(self._init_weights_) # Initialize weights (I need to understand this)
      self.last_layer      = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias = False))
      self.last_layer.weight_g.data.fill_(1) # I need to understand this better
      if norm_last_layer == True:
        self.last_layer.weight_g.requires_grad = False

    def _init_weights_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
      x = self.mlp(x)
      x = F.normalize(x, dim=-1, p=2)
      x = self.last_layer(x)
      return x


class MultiCropWrapper(nn.Module):
    # We are doing this for the explicit purpose of handling various crop sizes
    def __init__(
                self,
                backbone,
                proj_head
                ):
      super().__init__()
      backbone.fc , backbone.head = nn.Identity(), nn.Identity()
      self.backbone    =     backbone
      self.projector   =     proj_head

    def forward(self, x):
      if not isinstance(x, list): # I've converted them all to lists, but might as well assert
        x = [x]
      # The following line may be confusing, but just understand
      # this much that all it is doing is finding out the size of the images in x,
      # finding out how many unique sizes there are (say 2 global crops at 224, 8 local
      # crops at 96), finding out when these changes happen in the list x, and
      # then batching them separately
      sizes = torch.tensor([inp.shape[-1] for inp in x])
      unique_sizes, counts = torch.unique_consecutive(sizes, return_counts=True)
      idx_crops = torch.cumsum(counts, 0)

      start_idx, output = 0, torch.empty(0).to(x[0].device)

      for last_idx in idx_crops:
        out = self.backbone(torch.cat(x[start_idx:last_idx]))

        if isinstance(out, tuple):
          out = out[0]
        output = torch.cat([output, out])
        start_idx = last_idx


      return self.projector(output)

############
# Prenorm module, makes it cheaper

class PreNorm(nn.Module):
    def __init__(self, norm, model, dimension):
        super().__init__()
        self.norm = norm(dimension)
        self.model = model
    def forward(self,x):
        return self.model(self.norm(x))


#############

# Adding the LARS optimizer for large batch training when this is possible. This is just plain copy paste from Meta repo.


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])





