from torchvision import transforms


def train_transforms():

    return transforms.Compose(
        [
            # extract random crops and resize to img_resolution
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )


def val_transforms(dataset, normalize=True, resize=None, crop=None):

    return transforms.ToTensor()