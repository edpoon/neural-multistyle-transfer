"""Helper module for Pillow image and network tensor conversions
"""

import torchvision.transforms as transforms


def image_to_tensor(image, size):
    """Converts an image to a network tensor

    Args:
    size (int, int) : Tensor size after converison
    """
    loader = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor()])
    # add a fake dimension to match VGG19 dimensions
    return loader(image).unsqueeze(0)


def to_image(x, size):
    """Convert an output tensor from the network to a PIL image

    Args:
    size (int, int) : Image size after conversion
    """
    x = x.cpu().squeeze(0).clone()
    return transforms.ToPILImage()(x).resize(size)
