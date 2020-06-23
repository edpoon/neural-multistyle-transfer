from argparse import ArgumentParser
import numpy as np
import torch
import imageio
from PIL import Image
import torchvision.transforms as transforms
from tqdm.auto import trange
from utils import to_image, image_to_tensor
from model import NeuralStyle

CONTENT_LAYERS = ['conv_4']
STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def style_transfer_image(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = Image.open(args.content_target)
    size = image.size
    content_target = image_to_tensor(image, size).to(device)
    style_targets = [
        image_to_tensor(Image.open(image), size).to(device)
        for image in args.style_targets
    ]
    n = len(style_targets)
    style_weights = np.ones(
        n) / n if args.style_weights is None else args.style_weights
    input_image = content_target.clone().to(device)

    neural_style = NeuralStyle(content_layers=CONTENT_LAYERS,
                               style_layers=STYLE_LAYERS)
    neural_style.content_target = content_target
    neural_style.set_style_targets(style_targets, style_weights)

    output_image, _ = neural_style.transfer(input_image=input_image,
                                         epochs=args.epochs,
                                         style_weight=args.style_weight,
                                         content_weight=args.content_weight,
                                         verbose=args.verbose)

    to_image(output_image, size=size).save(args.output)


def style_transfer_video(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = transforms.ToPILImage()
    reader = imageio.get_reader(args.content_target)

    frames = [
        image_to_tensor(loader(reader.get_data(i)), (512, 512))
        for i in range(reader.count_frames())
    ]
    style_targets = [
        image_to_tensor(Image.open(image), (512, 512))
        for image in args.style_targets
    ]
    style_weights = np.linspace(0, 1, num=len(frames))

    neural_style = NeuralStyle(content_layers=CONTENT_LAYERS,
                               style_layers=STYLE_LAYERS).to(device)
    input_image = frames[0].to(device)
    outputs = []
    for i in trange(len(frames)):
        neural_style.content_target = frames[i].to(device)
        neural_style.set_style_targets(
            style_targets, [1 - style_weights[i], style_weights[i]])
        output_image = neural_style.transfer(
            input_image=input_image,
            epochs=args.epochs,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            verbose=args.verbose)
        # del frames[i]
        input_image = output_image.clone().to(device)
        outputs.append(output_image.to('cpu'))
        del output_image

    writer = imageio.get_writer('output.mp4',
                                fps=reader.get_meta_data()['fps'])
    shape = reader.get_data(0).shape[:2]
    outputs = [to_image(output, (shape[1], shape[0])) for output in outputs]

    for output in outputs:
        writer.append_data(np.asarray(output))
    writer.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('--content', dest='content_target', required=True)
    parser.add_argument('--styles',
                        dest='style_targets',
                        nargs='+',
                        required=True)
    parser.add_argument('--epochs', type=int, dest='epochs', default=1001)
    parser.add_argument('--output', dest='output', default='output.jpg')
    parser.add_argument('--content_weight',
                        type=int,
                        dest='content_weight',
                        default=1)
    parser.add_argument('--style_weight',
                        type=int,
                        dest='style_weight',
                        default=1e6)
    parser.add_argument('--style_weights',
                        nargs='+',
                        type=float,
                        dest='style_weights')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    filetype = args.content_target.split('.')
    if filetype[-1] in ['jpg', 'jpeg', 'png', 'tiff']:
        style_transfer_image(args)
    else:
        style_transfer_video(args)


if __name__ == '__main__':
    main()
