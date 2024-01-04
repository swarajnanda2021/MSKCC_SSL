from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import ImageOps

# For SimCLR, NNCLR, and other invariance based methods
class ContrastiveTransformations(object):
    def __init__(self, size=32, nviews=2, **kwargs):
        self.size = size
        self.nviews = nviews
        self.horizontal_flip = kwargs.get('horizontal_flip', True)
        self.resized_crop = kwargs.get('resized_crop', True)
        self.color_jitter = kwargs.get('color_jitter', True)
        self.random_grayscale = kwargs.get('random_grayscale', True)
        self.to_tensor = kwargs.get('to_tensor', True)
        self.normalize = kwargs.get('normalize', True)
        self.brightness = kwargs.get('brightness', 0.5)
        self.contrast = kwargs.get('contrast', 0.5)
        self.saturation = kwargs.get('saturation', 0.5)
        self.hue = kwargs.get('hue', 0.1)
        self.color_jitter_p = kwargs.get('color_jitter_p', 0.8)
        self.grayscale_p = kwargs.get('grayscale_p', 0.2)
        self.mean = kwargs.get('mean', (0.5,))
        self.std = kwargs.get('std', (0.5,))

    def __call__(self, x):
        transforms_list = []
        transforms_list.append(transforms.Resize((self.size, self.size)))
        if self.horizontal_flip:
            transforms_list.append(transforms.RandomHorizontalFlip())
        if self.resized_crop:
            transforms_list.append(transforms.RandomResizedCrop(self.size))
        if self.color_jitter:
            color_jitter_transform = transforms.ColorJitter(
                brightness=self.brightness, 
                contrast=self.contrast, 
                saturation=self.saturation, 
                hue=self.hue
            )
            transforms_list.append(transforms.RandomApply([color_jitter_transform], p=self.color_jitter_p))
        if self.random_grayscale:
            transforms_list.append(transforms.RandomGrayscale(p=self.grayscale_p))
        if self.to_tensor:
            transforms_list.append(transforms.ToTensor())
        if self.normalize:
            transforms_list.append(transforms.Normalize(self.mean, self.std))

        composed_transforms = transforms.Compose(transforms_list)
        return [composed_transforms(x) for _ in range(self.nviews)]


# For DiNO based non-invariance based methods:


# The original DinoTransforms wasn't great, the right version has 2 global crops and n (8 in OG implementation) local crops

class DinoTransforms(object):


    def __init__(self,
                 local_size         = 96,
                 global_size        = 224,
                 local_crop_scale   = (0.05, 0.4),
                 global_crop_scale  = (0.4, 1.0),
                 n_local_crops      = 2,
                 ):

        self.n_local_crops = n_local_crops

        # Assign some canonical transformations
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=global_size, scale = global_crop_scale),
            transforms.RandomApply([flip_and_color_jitter], p=0.8),
            transforms.GaussianBlur(3, (0.1,0.15)),
            normalize,
        ])

        self.global_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=global_size, scale = global_crop_scale),
            transforms.RandomApply([flip_and_color_jitter], p=0.8),
            transforms.GaussianBlur(kernel_size = 3, sigma = (0.1,0.15)),
            transforms.Lambda(lambda img: ImageOps.solarize(img, threshold=128)),
            normalize,
        ])

        self.local    = transforms.Compose([
            transforms.RandomResizedCrop(size=local_size, scale = local_crop_scale),
            normalize,
        ])


    def __call__(self,x):
        crops = []

        crops.append(self.global_1(x))
        crops.append(self.global_2(x))
        for _ in range(self.n_local_crops):
            crops.append(self.local(x))

        return crops





# Now place the usage example in a if __name__ == "__main__": block
if __name__ == "__main__":
    contrastive == 0
    # Initialize the ContrastiveTransformations object with custom parameters
    if contrastive == 1:
        custom_transforms = ContrastiveTransformations(
            size=32,
            nviews=2,
            horizontal_flip=True,
            resized_crop=True,
            color_jitter=True,
            random_grayscale=True,
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.1,
            color_jitter_p=0.8,
            grayscale_p=0.2,
            to_tensor=True,
            normalize=True,
            mean=(0.5,),
            std=(0.5,)
        )
        
        cifar_trainset  = CIFAR10(root='./data',train=True,download=True, transform=contrastive_transform)
    else:
        custom_transforms = DinoTransforms(
                                  local_size         = 96,
                                  global_size        = 224,
                                  local_crop_scale   = (0.05, 0.4),
                                  global_crop_scale  = (0.4, 1.0),
                                  n_local_crops      = CROPS,
                                  )

    
    # If you are working with the CIFAR10 dataset (for implementation), use the following line
    cifar_trainset = CIFAR10(root='./data', train= True, download=True, transform=custom_transforms)

    # If you have your own image folder, then use the torvision.datasets.ImageFolder function
    image_dataset = ImageFolder(root='root_directory', transform=custom_transforms)



