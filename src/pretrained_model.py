import sys
sys.path.append("./PyTorch_CIFAR10/")
import cifar10_models.vgg as cifar10_vgg
import resnet_model_1
import resnet_model_2
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
    else:
        raise Exception("model name or dataset error")