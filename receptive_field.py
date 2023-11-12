import numpy as np
import torch
from matplotlib import pyplot as plt


if __name__ == '__main__':



    from models.resnet import resnet50
    model = resnet50(num_classes=10)

    # from models.GIB_resnet import GIB_resnet50
    # #
    # model = GIB_resnet50(num_classes=10)



    # All parameters are set to 0.1
    for param in model.parameters():
        param.data.fill_(0.1)

    model.eval()

    # Record the gradient
    dAdi_np_cum = np.zeros((224, 224))


    def try_gpu(i=0):
        '''Change to GPU'''
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')


    device = try_gpu(1)
    model.to(device)

    # Generate an image with a value of 1
    input = torch.ones((1, 3, 224, 224))
    input = input.to(device)


    # hooker funciton
    def save_gradient(grad):
        return grad


    feature_dict = {}

    # img도 hook
    x = input
    x.requires_grad = True
    x.register_hook(save_gradient)
    feature_dict["img"] = x

    # for ResNet
    for name, module in model._modules.items():
        # if name != "fc":
        if name not in ["fc", "classifier"]:
            x = module(x)
            x.register_hook(save_gradient)
            feature_dict[name] = x

    # Stage1= layer1, Stage2= layer2 ...
    # A_ijk = feature_dict["layer4"]  # 1, 512, 7, 7
    # A_ijk = feature_dict["layer3"]
    # A_ijk = feature_dict["layer2"]
    A_ijk = feature_dict["layer1"]

    # center point
    # A = A_ijk[0, :, 3, 3].mean()  # channel mean, center:(3, 3).
    # A = A_ijk[0, :, 7, 7].mean()
    # A = A_ijk[0, :, 14, 14].mean()
    A = A_ijk[0, :, 28, 28].mean()
    A.backward(retain_graph=True)

    dAdi = input.grad  # 1, 3, 224, 224
    dAdi_np = dAdi.cpu().data.numpy()
    ## dAdi_np = dAdi_np.sum(axis=(0, 1))
    dAdi_np = dAdi_np.mean(axis=(0, 1))  # channel mean
    dAdi_np = np.maximum(dAdi_np, 0)  # 224, 224
    dAdi_np_cum += dAdi_np

    # Get the index with gradient greater than 0
    index = np.nonzero(dAdi_np_cum > 0.0)

    # We do not compare the gradient magnitude, only need to see
    # if any gradient arrived, so all are set to 1
    for i, j in zip(index[0], index[1]):
        dAdi_np_cum[i][j] = 1


    # Plotting gradients
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = plt.axes()
    plt.imshow(dAdi_np_cum, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')

    x_ticks = np.arange(0, 224, 28)  # Set the X-axis scale with a step size of 28
    y_ticks = x_ticks

    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # plt.savefig("RF-224-layer1.png", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    # plt.savefig("RF-GIB-224-layer1.png", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.show()
