import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import seaborn as sns
from .cams import *
from .utils import *
from ..model import *
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image

def load_configs():
    RESNET_CONFIG = {_resnet: dict(input_layer='conv1', conv_layer='layer4', fc_layer='linear') for _resnet in resnet.__dict__.keys()}
    VGG_CONFIG = {_vgg: dict(input_layer='features', conv_layer='features') for _vgg in vgg.__dict__.keys()}
    MODEL_CONFIG = {**RESNET_CONFIG,**VGG_CONFIG}
    return MODEL_CONFIG
                

def load_cams(model, device,model_name):
    model.to(device)
    model.eval()
    MODEL_CONFIG = load_configs()
    if 'ResNet18' == model_name:
        conv_layer = MODEL_CONFIG['ResNet18']['conv_layer']
        input_layer = MODEL_CONFIG['ResNet18']['input_layer']
        fc_layer = MODEL_CONFIG['ResNet18']['fc_layer']
    cam_extractors = [CAM(model, conv_layer, fc_layer), GradCAM(model, conv_layer),GradCAMpp(model, conv_layer),
                     SmoothGradCAMpp(model, conv_layer, input_layer)]
    return cam_extractors

def plot_cams(images,labels,cam_extractors,model,std,mean,class_labels):
    
    fig, axes = plt.subplots(len(images), len(cam_extractors)+1, figsize=(100,100))
    
    for _ in range(len(images)):
        de_image = denormalize(images[_],std,mean)
        # axes[_][0].imshow(to_pil_image(images[_].cpu()))
        axes[_][0].imshow(to_pil_image(de_image.cpu()))
        for idx, extractor in enumerate(cam_extractors):
            model.zero_grad()
            # scores = model(img_tensor.unsqueeze(0))
            # scores = model(images[_].unsqueeze(0))
            scores = model(de_image.unsqueeze(0))

            # Select the class index
            class_idx = scores.squeeze(0).argmax().item()

            # Use the hooked data to compute activation map
            activation_map = extractor(class_idx, scores).cpu()
            # Clean data
            extractor.clear_hooks()
            # Convert it to PIL image
            # The indexing below means first image in batch
            heatmap = to_pil_image(activation_map, mode='F')
            # Plot the result
            # result = overlay_mask(to_pil_image(images[_].cpu()), heatmap)
            result = overlay_mask(to_pil_image(de_image.cpu()), heatmap)

            axes[_][idx+1].imshow(result)
            axes[_][idx+1].axis('off')
            axes[_][idx+1].set_title(extractor.__class__.__name__, size=50)
        axes[_][0].set_title("Actual:"+str(class_labels[labels[_]])+"\nPredicted:" + str(class_labels[class_idx]), size=50)
        # axes[7].imshow(to_pil_image(images[_].cpu())
        # axes[7].axis('off')
        # axes[7].set_title(labels[_])
        
    plt.tight_layout()
    plt.show()
