from argparse import ArgumentParser
from PIL import Image
import torch
from utils import load_image, to_image
from model import NST

def main():
    parser = ArgumentParser()
    parser.add_argument('--content',
                        dest='content_image',
                        metavar='CONTENT',
                        required=True)
    parser.add_argument('--styles',
                       dest='style_images',
                       nargs='+',
                       metavar='STYLES',
                       required=True)
    parser.add_argument('--style_weights',
                       dest='style_weights',
                       metavar='STYLE_WEIGHTS')
    args = parser.parse_args()
     
    nst = NST()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size = (512, 512) if torch.cuda.is_available() else (128, 128)
    content_target = load_image(file_name=args.content_image, size=size).to(device)
    style_targets = [load_image(file_name=name, size=size).to(device) for name in args.style_images]
    nst = NST()
    nst.content_target = content_target
    
    style_weights = [1 / len(style_targets)] * len(style_targets)
    nst.set_style_targets(style_targets, style_weights)
    STYLE_WEIGHT = 1000000
    CONTENT_WEIGHT = 1
    EPOCHS = 1001
    input_image = torch.rand(nst.content_target.shape).to(device)
    output_image, _ = nst.style_transfer(input_image=input_image,
                                         epochs=EPOCHS,
                                         style_weight=STYLE_WEIGHT,
                                         content_weight=CONTENT_WEIGHT,
                                         silent=True)
    to_image(output_image).save('examples/outputs/output.jpg')

if __name__ == '__main__':
    main()
