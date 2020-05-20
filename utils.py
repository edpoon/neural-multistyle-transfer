import torchvision.transforms as transforms
from PIL import Image
import imageio

def image_to_tensor(image, size):
    loader = transforms.Compose([transforms.Resize(size),
                                 transforms.ToTensor()])
    # add a fake dimension to match VGG19 dimensions
    return loader(image).unsqueeze(0)

def to_image(x, size):
    '''Convert an output tensor from the network to a PIL image
    '''
    x = x.cpu().squeeze(0).clone()
    return transforms.ToPILImage()(x).resize(size)
