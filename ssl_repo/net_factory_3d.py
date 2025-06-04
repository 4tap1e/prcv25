from .networks import unet_3D
from .networks import VNet
# from .networks import Attention_UNet
from .nnunet import initialize_network    


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    # elif net_type == "attention_unet":
    #     net = Attention_UNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    else:
        net = None
    return net
