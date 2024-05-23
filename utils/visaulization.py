import gc
from typing import Callable
from PIL import Image
from matplotlib import cm
import numpy as np

import torch


class VGGCamExtractor(object):
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        with torch.autocast(device_type='cuda', enabled=False, dtype=torch.float32):
            conv_output, x = self.forward_pass_on_convolutions(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return conv_output, x


class VGGGradCam(object):
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.model.to('cpu')
        self.extractor = VGGCamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        conv_output, model_output = self.extractor.forward_pass(input_image.to('cpu'))
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to('cpu')
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        # cam = np.resize(cam, (input_image.shape[2], input_image.shape[3]))

        x = torch.tensor(cam.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        output = torch.nn.functional.interpolate(input=x, size=(input_image.shape[2], input_image.shape[3]),
                                                 mode="bilinear", align_corners=True)
        cam = output[0, 0].numpy()

        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        return cam


def save_class_activation_on_image(org_img, activation_map):

    activation_map_image = Image.fromarray(np.uint8(cm.hsv(activation_map) * 255)).convert('RGB')

    blended = Image.blend(org_img, activation_map_image, .5)

    return np.concatenate([np.asarray(org_img), np.asarray(activation_map_image), np.asarray(blended)], axis=1)


class GradCAMReporter(object):
    def __init__(self, data_pack_x_clear, data_pack_x_preprocessed, data_pack_y,
                 layers, grad_cam_class: Callable = VGGGradCam):
        self.data_pack_x_clear = data_pack_x_clear
        self.data_pack_x_preprocessed = data_pack_x_preprocessed
        self.data_pack_y = data_pack_y

        self.layers = layers
        self.grad_cam_class = grad_cam_class

    def create_report(self, model):

        output_layers = {}
        for _layer in self.layers:
            gradcam = self.grad_cam_class(model, _layer)
            inner_list = []
            _layer_key = _layer if isinstance(_layer, str) else f"layer_{_layer}"
            for i in range(self.data_pack_x_preprocessed.shape[0]):
                cam = gradcam.generate_cam(self.data_pack_x_preprocessed[i:i+1], self.data_pack_y[i])
                inner_list.append(save_class_activation_on_image(self.data_pack_x_clear[i], cam))
            output_layers[_layer_key] = Image.fromarray(np.concatenate(inner_list, axis=0))

            del gradcam
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return output_layers
