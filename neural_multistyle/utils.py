"""General utility functions"""

import json
from typing import Tuple
from torch import Tensor
import torchvision.transforms as transforms


def image_to_tensor(image: Tensor, size: Tuple[int, int]) -> Tensor:
    """Converts image to a torch tensor for network input

    Args:
        image: Input image
        size: Tensor size after converison
    """
    loader = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    image_tensor = loader(image)
    # add a fake dimension to match VGG19 dimensions
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def to_image(x: Tensor, size: Tuple[int, int]):
    """Convert an output tensor from the network to a PIL image

    Args:
        x: Input tensor
        size: Image size after conversion
    """
    transform = transforms.ToPILImage()
    # remove fake dimension used to match VGG19 dimensions
    x = x.cpu().squeeze(0).clone()
    image = transform(x).resize(size)
    return image


class Params:
    """Class that loads hyperparameters from a json file

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params = learning_rate = 0.5 # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path: str):
        self.update(json_path)

    def save(self, json_path: str):
        """Save parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path: str):
        """Load parameters with json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

    def __repr__(self):
        return repr(self.__dict__)
