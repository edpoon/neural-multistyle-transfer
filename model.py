import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gram_matrix(x):
    """Computes the gram matrix of the input

    Parameters
    ----------
    input : tensor
    The tensor to compute the gram matrix
    """
    a, b, c, d = x.size()
    features = x.view(a * b, c * d)
    # For input matrices with large dimensions, the gram matrix
    # typically has large values, thus we normalize
    return (features @ features.T) / (a * b * c * d)

class Normalization(torch.nn.Module):
    '''Normalization layer

    Normalizes the input tensor to  match the model
    That is, every value x in the tensor is mapped to the standard score

    Attributes
    ----------
    mean : tensor
    The mean of the normalization
    std : tensor
    The standard deviation of the normalization
    Methods
    -------
    forward(x):
    Normalizes the input the this layer
    '''
    def __init__(self, means, stds):
        super(Normalization, self).__init__()
        self.mean = torch.tensor([[[mean]] for mean in means]).to(device)
        self.std = torch.tensor([[[std]] for std in stds]).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class ContentLoss(torch.nn.Module):
    '''A dummy layer to compute the content loss at a given layer

    Attributes
    ----------
    target : tensor
    The hidden representation of the content image at this layer
    loss : num
    The content loss between the hidden representations of the content image
    and the input image at this layer
    '''
    def __init__(self):
        super(ContentLoss, self).__init__()
        # self.target = target.detach()
        self.target = None

    def forward(self, x):
        if self.target is not None:
            self.loss = F.mse_loss(x, self.target)
        return x

class StyleLoss(torch.nn.Module):
    '''A dummy layer to compute the style loss at a given layer

    Attributes
    ----------
    target : [tensor]
    The hidden representations of the style images at this layer
    loss : num
    The content loss between the hidden representations of the style images
    and the input image at this layer
    '''
    def __init__(self):
        super(StyleLoss, self).__init__()
        # self.target = gram_matrix(target).detach()
        self.target = None

    def forward(self, x):
        if self.target is not None:
            self.loss = F.mse_loss(gram_matrix(x), self.target)
        return x

class NST(object):
    """Runs the multistyle transfer algorithms

  Attributes
    ----------
    content_layers : [str]
    A list of layers that contributes to the total content loss
    style_layers : [str]
    A list of layers that contributes to the total style loss
    weights : [int]
    For multistyle transfer, each value corresponds to the weight of that style
    in the total loss function
    style_images : [tensor]
    The style images for which we do multistyle transfer from
    content_image : tensor
    The content image for which we apply the mutlistyle transfer to

  Methods
    -------
    build_model(net, content_layers, style_layers)
    Builds a model for neural style transfer from VGG19
    calculate_loss(content_layers, style_layers)
    Calculates the total loss of an input image
    style_transfer_gd()
    Runs style transfer via repeatedly doing feedforward and backpropogation
    via gradient descent. This currently doesn't work
    style_transfer_lbfgs()
    Runs style transfer via repeatedly doing feedforward and backpropogation
    via lbfgs
    """
    def __init__(self, content_layers, style_layers, style_image, content_image):
        self.model, self.content_layers, self.style_layers = self.init_model(content_layers, style_layers)

    @property
    def style_target(self):
        return self._style_target

    @style_target.setter
    def style_target(self, target):
        with torch.no_grad():
            self._style_target = target
            for layer in self.model.children():
                target = layer(target)
                if isinstance(layer, StyleLoss):
                    layer.target = gram_matrix(target)

    @property
    def content_target(self):
        return self._content_target

    @content_target.setter
    def content_target(self, target):
        with torch.no_grad():
            self._content_target = target
            for layer in self.model.children():
                target = layer(target)
                if isinstance(layer, ContentLoss):
                    layer.target = target

    def init_model(self, content_layers, style_layers):
        '''Builds the model for multistyle neural style transfer from vgg19

        Parameters
        m
        content_layers : [str]
        A list of layers that contributes to the total content loss
        style_layers : [str]
        A list of layers that contributes to the total style loss
        '''
        # import vgg19 with the pretrained weights
        net = models.vgg19(pretrained=True).features.to(device).eval()
        # start our sequential model whose first layer in normalization
        model = torch.nn.Sequential()
        model.add_module('norm_0', Normalization(means=[0.485, 0.456, 0.406],
                                                 stds=[0.229, 0.224, 0.225]))
        # keep track of the layers to compute style loss and content loss
        style_losses, content_losses = [], []
        i = 0
        for layer in net.children():
            if isinstance(layer, torch.nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, torch.nn.ReLU):
                name = f'relu_{i}'
                # the ReLU layer in the vgg19 implementation is done in place
                # which will throw an error for autograd during backpropogation
                layer = torch.nn.ReLU()
            elif isinstance(layer, torch.nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, torch.nn.BatchNorm2d):
                name = f'bn_{i}'

            model.add_module(name, layer)

            # add dummy content loss layer
            if name in content_layers:
                # target = model(self._content_target).detach()
                layer = ContentLoss()
                model.add_module(f'content_loss_{i}', layer)
                content_losses.append(layer)

            # add dummy style loss layer
            if name in style_layers:
                # target = model(self._style_target).detach()
                layer = StyleLoss()
                model.add_module(f'style_loss_{i}', layer)
                style_losses.append(layer)

        # trim the remaining layers proceeding the last content loss or style loss layer
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        return model[:i+1], content_losses, style_losses

    def calculate_losses(self):
        '''Calculates the total loss of an input image
        '''
        content_loss = sum([loss.loss for loss in self.content_layers]) / len(self.content_layers)
        style_loss = sum([loss.loss for loss in self.style_layers]) / len(self.style_layers)

        return content_loss, style_loss

    def style_transfer(self, input_image, epochs=300, style_weight=1000000, content_weight=1, silent=True):
        '''Runs style transfer via repeatedly doing feedforward and backpropogation via lbfgs

        Parameters
        ----------
        input_image : tensor
        The initial input image to be optimized.
        Example: white noise, content image, style image, or any image.
        epochs : int
        The number of epochs to run the lbfps algorithm for
        style_weight : num
        The weight of the total style loss function in the total loss function
        content_weight : num
        The weight of the total content loss function in the total loss function
        '''
        # make a copy of the input_image so that we don't modify the original image
        input_img = input_image.clone().to(device)
        optimizer = optim.LBFGS([input_img.requires_grad_()])

        losses = {'style_loss': [], 'content_loss': [], 'total_loss': []}

        if not silent:
            # calculate initial loss, print and store them
            self.model(input_img)
            content_loss, style_loss = self.calculate_losses()
            total_loss = style_weight * style_loss + content_weight * content_loss

            print(f'Initial style Loss : {style_loss.item()} Initial content Loss: {content_loss.item()}')
            print(f'Initial total Loss : {total_loss.item()}')
            losses['style_loss'].append(style_loss.item())
            losses['content_loss'].append(content_loss.item())
            losses['total_loss'].append(total_loss.item())

        run = [0]
        # the value of run[0] here means we have finished that value of epochs
        while run[0] < epochs:
            def closure():
                input_image.data.clamp_(0, 1)
                optimizer.zero_grad()

                self.model(input_img)
                content_loss, style_loss = self.calculate_losses()
                total_loss = style_weight * style_loss + content_weight * content_loss
                total_loss.backward()

                if not silent:
                    losses['style_loss'].append(style_loss.item())
                    losses['content_loss'].append(content_loss.item())
                    losses['total_loss'].append(total_loss.item())

                # increase by 1 when we have finished this epoch
                run[0] += 1

                if not silent:
                    # print losses every print_epoch epochs
                    print_epoch = 10 # Change this if you want!

                    if run[0] % print_epoch == 0:
                        print("epoch {}:".format(run))
                        print(f'style Loss : {style_loss.item()} content Loss: {content_loss.item()}')
                        print(f'total Loss : {total_loss.item()}')
                return total_loss

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)

        if not silent:
            self.model(input_img)
            content_loss, style_loss = self.calculate_losses()
            total_loss = style_weight * style_loss + content_weight * content_loss
            print(f'Final style Loss : {style_loss.item()} content Loss: {content_loss.item()}')
            print(f'Final total Loss : {total_loss.item()}')
            losses['style_loss'].append(style_loss.item())
            losses['content_loss'].append(content_loss.item())
            losses['total_loss'].append(total_loss.item())

        return input_img if silent else input_img, losses

    def __str__(self):
        return str(self.model)
