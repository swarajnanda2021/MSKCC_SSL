from torchvision import transforms


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


# Now place the usage example in a if __name__ == "__main__": block
if __name__ == "__main__":
    # Initialize the ContrastiveTransformations object with custom parameters
    contrastive_transform = ContrastiveTransformations(
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
