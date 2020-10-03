import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from .cam import GradCAM


# def load_gradcam(images, labels, model, device, target_layers):
def load_gradcam(test, model, device, target_layers,size = 25,classified = True):

    _images = []
    _target = []
    _pred = []

    # model, device = self.trainer.model, self.trainer.device

    # set the model to evaluation mode
    model.eval()

    # turn off gradients
    with torch.no_grad():
        for data, target in test:
            # move them to respective device
            data, target = data.to(device), target.to(device)

            # do inferencing
            output = model(data)
            # print("output:",output[0])
            # get the predicted output
            pred = output.argmax(dim=1, keepdim=True)
            # print(pred,pred.view_as(target))

            # get the current misclassified in this batch
            list_images = (target.eq(pred.view_as(target)) == classified)
            batch_misclassified = data[list_images]
            batch_mis_pred = pred[list_images]
            batch_mis_target = target[list_images]

            # batch_misclassified =

            _images.append(batch_misclassified)
            _pred.append(batch_mis_pred)
            _target.append(batch_mis_target)

    # group all the batched together
    img = torch.cat(_images)
    pred = torch.cat(_pred)
    tar = torch.cat(_target)
    # move the model to device

    images = img[:size]
    labels = tar[:size]

    model.to(device)

    # set the model in evaluation mode
    model.eval()

    # get the grad cam
    gcam = GradCAM(model=model, candidate_layers=target_layers)

    # images = torch.stack(images).to(device)

    # predicted probabilities and class ids
    pred_probs, pred_ids = gcam.forward(images)

    # actual class ids
    # target_ids = torch.LongTensor(labels).view(len(images), -1).to(device)
    target_ids = labels.view(len(images), -1).to(device)

    # backward pass wrt to the actual ids
    gcam.backward(ids=target_ids)

    # we will store the layers and correspondings images activations here
    layers_region = {}

    # fetch the grad cam layers of all the images
    for target_layer in target_layers:

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        layers_region[target_layer] = regions

    # we are done here, remove the hooks
    gcam.remove_hook()

    return layers_region, pred_probs, pred_ids,images, labels


sns.set()
# plt.style.use("dark_background")


def plot_gradcam(gcam_layers, images, target_labels, predicted_labels, class_labels, denormalize):

    images = images.cpu()
    # convert BCHW to BHWC for plotting stufffff
    images = images.permute(0, 2, 3, 1)
    target_labels = target_labels.cpu()

    fig, axs = plt.subplots(nrows=len(images), ncols=len(
        gcam_layers.keys())+1, figsize=((len(gcam_layers.keys()) + 2)*3, len(images)*3))
    fig.suptitle("Grad-CAM", fontsize=16)

    for image_idx, image in enumerate(images):

        # denormalize the imaeg
        denorm_img = denormalize(image.permute(2, 0, 1)).permute(1, 2, 0)

        # axs[image_idx, 0].text(
        #     0.5, 0.5, f'predicted: {class_labels[predicted_labels[image_idx][0] ]}\nactual: {class_labels[target_labels[image_idx]] }', horizontalalignment='center', verticalalignment='center', fontsize=14, )
        # axs[image_idx, 0].axis('off')

        axs[image_idx, 0].imshow(
            (denorm_img.numpy() * 255).astype(np.uint8),  interpolation='bilinear')
        axs[image_idx, 0].axis('off')

        for layer_idx, layer_name in enumerate(gcam_layers.keys()):
            # gets H X W of the cam layer
            _layer = gcam_layers[layer_name][image_idx].cpu().numpy()[0]
            heatmap = 1 - _layer
            heatmap = np.uint8(255 * heatmap)
            heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed_img = cv2.addWeighted(
                (denorm_img.numpy() * 255).astype(np.uint8), 0.6, heatmap_img, 0.4, 0)

            axs[image_idx, layer_idx +
                1].imshow(superimposed_img, interpolation='bilinear')
            axs[image_idx, layer_idx+1].set_title(f'layer: {layer_name}')
            axs[image_idx, layer_idx+1].axis('off')
        axs[image_idx, 0].set_title(f'Predicted: {class_labels[predicted_labels[image_idx][0] ]}\nTarget: {class_labels[target_labels[image_idx]] }')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, wspace=0.2, hspace=0.2)
    plt.show()