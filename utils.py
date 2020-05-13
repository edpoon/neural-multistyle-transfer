import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

def load_image(file_name, size):
    '''Loads an image and returns it as a tensor

    Parameters
    ----------
    file_name : str
        path to the image
    size : (int, int)
        size of the image to reshape to
    '''
    loader = transforms.Compose([transforms.Resize(size),
                                 transforms.ToTensor()])
    image = Image.open(file_name)
    # add a fake dimension to match VGG19 dimensions
    tensor = loader(image).unsqueeze(0)
    tensor.image_name = file_name.split('.')[0]
    return tensor

def to_image(tensor):
    '''Convert tensors from the NN to Pillow image
    '''
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    loader = transforms.ToPILImage()
    return loader(image)
