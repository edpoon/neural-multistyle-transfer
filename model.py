import torch
import torch.nn.functional as F
import torchvision.models as models

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    return (features @ features.T) / (a * b * c * d)

class Normalization(torch.nn.Module):
    def __init__(self,
                 mean=[[[0.485]], [[0.456]], [[0.406]]],
                 std=[[[0.229]], [[0.224]], [[0.225]]]):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, tensor):
        return (image - self.mean) / self.std

class ContentLoss(torch.nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(torch.nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, input):
        self.loss = F.mse_loss(gram_matrix(input), target)
        return input

class Net(torch.nn.Module):
    def __init__(self, size, content_layers, style_layers, content_image, style_image):
        super(Net, self).__init__()
        self.model, self.content_losses, self.style_losses = self.__init_model__(content_layers,
                                                                                style_layers,
                                                                                content_image,
                                                                                style_image)


    def __init_model__(self, content_layers, style_layers, content_image, style_image):
        net = models.vgg19(pretrained=True).features.eval()
        model = torch.nn.Sequential()
        # keep track of the layers to compute the style loss and content loss
        style_losses = []
        content_losses = []

        model.add_module('norm_{i}', Normalization())
        i = 0
        for layer in net.children():
            if isinstance(layer, torch.nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, torch.nn.ReLU):
                name = f'relu_i'
                # the ReLU layers in the vgg19 implemention are done in place
                # which will throw an error for autograd during backpropogation
                layer = torch.nn.ReLU()
            elif isinstance(layer, torch.nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, torch.nn.BatchNorm2d):
                name = f'bn_{i}'

            model.add_module(name, layer)

            if name in content_layers:
                target = net(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module(f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target = net(style_image).detach()
                style_loss = StyleLoss(target)
                model.add_module(f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)

        # trim layers proceeding the last content loss or style loss layer
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        return model[:i+1], content_losses, style_losses
