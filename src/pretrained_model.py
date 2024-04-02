import sys
import PyTorch_CIFAR10.cifar10_models.vgg as cifar10_vgg
import resnet_model_1
import resnet_model_2
from mobilevit_v2 import MobileViTv2
from options.opts import get_training_arguments

import torchvision.models

def get_pretrained_model(model_name, dataset):
    if(model_name == "vgg19_bn" and dataset == "cifar10"):
        return cifar10_vgg.vgg19_bn(pretrained = True)
    elif(model_name == "vgg19_bn" and dataset == "imagenet_1k"):
        return torchvision.models.vgg19_bn(weights="IMAGENET1K_V1")
    elif(model_name == "resnet18" and dataset == "imagenet_1k"):
        return resnet_model_1.resnet18_fp(pretrained= True)
    elif(model_name == "resnet32" and dataset == "cifar100"):
        return resnet_model_2.cifar100_resnet32(pretrained= True)
    elif(model_name == "resnet20" and dataset == "cifar10"):
        return resnet_model_2.cifar10_resnet20(pretrained = True)
    elif(model_name == "mobileVitV2" and dataset == "imagenet_1k"):
        args_list = ['--common.config-file', '/home/jianming/work/Fast_Switch/NN_Model/ml-cvnets/config/classification/imagenet/mobilevit_v2.yaml', '--common.results-loc', 'mobilevitv2_results/width_0_5_0', '--model.classification.pretrained', '/home/jianming/work/Fast_Switch/NN_Model/ml-cvnets/mobilevitv2_results/width_0_5_0/mobilevitv2-0.5.pt', '--common.override-kwargs', 'model.classification.mitv2.width_multiplier=0.5']
        opts = get_training_arguments(parse_args=True, args=args_list)
        model = MobileViTv2(opts)
        return model
    else:
        raise Exception("model name or dataset error")