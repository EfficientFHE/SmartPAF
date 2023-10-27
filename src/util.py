import os
import math
import copy
import sys
import yaml

from argparse import ArgumentParser
from typing import Any, Dict, Tuple, Union

try:
	from urllib import urlretrieve
except ImportError:
	from urllib.request import urlretrieve

import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.transforms import transforms
from torchvision import datasets










def load_model_yaml(path, config_name):
    with open(f"{path}{config_name}", 'r') as file:
        return yaml.safe_load(file)




def replace_layer(model, old_name, new_layer):
    tokens = old_name.split(".")
    last_token = tokens.pop(len(tokens) - 1)
    for token in tokens:
        if(token.isnumeric()):
            model = model[int(token)]
        else:
            model = getattr(model, token)
    setattr(model, last_token, new_layer)


def access_layer(model, name):
    tokens = name.split(".")
    for token in tokens:
        if(token.isnumeric()):
            model = model[int(token)]
        else:
            model = getattr(model, token)
    return model


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                print(str(key_item_1[0]) + " x " + str(key_item_2[0]))
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def generate_sign_nest_dict(model: nn.modules):
    sign_nest_dict= {}
    weight = []
    for name, layer in model.named_modules():
        class_name = layer.__class__.__name__
        if "MaxPool2d" in class_name:
            maxpool_dict = {"name": str, "type": str, "kernel_size" : Union[int, Tuple[int, int]], "stride": Union[int, Tuple[int, int]], "padding" : Union[int, Tuple[int, int]], "dilation": Union[int, Tuple[int, int]]}
            maxpool_dict["name"] = name
            maxpool_dict["type"] = "MaxPool2d"
            layer = access_layer(model,name)
            maxpool_dict["kernel_size"] = layer.kernel_size
            maxpool_dict["stride"] = layer.stride
            maxpool_dict["padding"] = layer.padding
            maxpool_dict["dilation"] = layer.dilation
            maxpool_dict["up_weight"] = weight
            weight = []
            sign_nest_dict[name] = maxpool_dict
        elif "ReLU" in class_name:
            relu_dict = {"name" : str, "type": str}
            relu_dict["name"] = name
            relu_dict["type"] = "ReLU"
            relu_dict["up_weight"] = weight
            weight = []
            sign_nest_dict[name] = relu_dict
        elif("Conv" in class_name) or ("BatchNorm" in class_name) or ("Linear" in class_name):
            weight.append(name)
    return sign_nest_dict







def train_data_loader_imagenet(data_dir = None, batch_size = 40, num_workers = 12):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset =  datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    return dataloader

def valid_data_loader_imagenet(data_dir = None, batch_size = 40, num_workers = 12):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset =  datasets.ImageFolder(
            root=os.path.join(data_dir, 'val'),
            transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return dataloader



def train_data_loader_cifar10(data_dir =None, batch_size = 100, num_workers = 12, download = False):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=download)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader



def valid_data_loader_cifar10(data_dir = None, batch_size = 100, num_workers = 12, download = False):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=download)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return dataloader

def train_data_loader_cifar100(data_dir =None, batch_size = 100, num_workers = 12, download = False):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        dataset = datasets.CIFAR100(root=data_dir, train=True, transform=transform, download=download)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader



def valid_data_loader_cifar100(data_dir = None, batch_size = 100, num_workers = 12, download = False):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = datasets.CIFAR100(root=data_dir, train=False, transform=transform, download=download)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return dataloader


def get_data_loader(dataset:str, dataset_type:str, data_dir,  batch_size = None, num_workers = None):
    dataset_nest_dict = {"cifar10":      {"batch_size":100,  "num_workers":12},
                          "imagenet_1k":   {"batch_size":40,   "num_workers":12},
                          "cifar100":      {"batch_size":100,  "num_workers":12}}
    if(dataset == "cifar10"):
        if(dataset_type == "train"):
            return train_data_loader_cifar10(data_dir, 
                                           (batch_size, dataset_nest_dict[dataset]["batch_size"]) [batch_size == None], 
                                           (batch_size, dataset_nest_dict[dataset]["num_workers"]) [num_workers == None])
        elif(dataset_type == "valid"):
            return valid_data_loader_cifar10(data_dir, 
                                          (batch_size, dataset_nest_dict[dataset]["batch_size"]) [batch_size == None], 
                                          (batch_size, dataset_nest_dict[dataset]["num_workers"]) [num_workers== None])
        else:
            raise Exception("dataset type error.")
    elif(dataset == "cifar100") :
        if(dataset_type == "train"):
            return train_data_loader_cifar100(data_dir, 
                                           (batch_size, dataset_nest_dict[dataset]["batch_size"]) [batch_size == None], 
                                           (batch_size, dataset_nest_dict[dataset]["num_workers"]) [num_workers == None])
        elif(dataset_type == "valid"):
            return valid_data_loader_cifar100(data_dir, 
                                          (batch_size, dataset_nest_dict[dataset]["batch_size"]) [batch_size == None], 
                                          (batch_size, dataset_nest_dict[dataset]["num_workers"]) [num_workers== None])
        else:
            raise Exception("dataset type error.")
         
    elif(dataset == "imagenet_1k"):
        if(dataset_type == "train"):
            return train_data_loader_imagenet(data_dir, 
                                           (batch_size, dataset_nest_dict[dataset]["batch_size"]) [batch_size == None], 
                                           (batch_size, dataset_nest_dict[dataset]["num_workers"]) [num_workers == None])
        elif(dataset_type == "valid"):
            return valid_data_loader_imagenet(data_dir, 
                                          (batch_size, dataset_nest_dict[dataset]["batch_size"]) [batch_size == None], 
                                          (batch_size, dataset_nest_dict[dataset]["num_workers"]) [num_workers == None])
        else:
            raise Exception("dataset type error.")
    else:
        raise Exception("dataset error.")




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str = " ", fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.reshape(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res




def validate(net, data_loader, device='cuda:0'):
    if 'cuda' in device:
        net = torch.nn.DataParallel(net).to(device)
    else:
        net = net.to(device)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)

    net.eval()
    net = net.to(device)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            # compute output
            output = net(images)
            loss = criterion(output, labels)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

    
    print('Results: loss=%.5f,\t top1=%.4f,\t top5=%.4f' % (losses.avg, top1.avg, top5.avg))
    return losses.avg, top1.avg, top5.avg


class Sign_parameter_generator():
    def __init__(self):
        self.param_nest_dict = {}
        
        #a7
        param_dict= {"name":str, "coef": torch.Tensor, "degree": list, "scale" : Union[int, torch.Tensor]}
        param_dict["name"] = "a7"
        coeflist = [[7.30445164958251,  -3.46825871108659*10,  5.98596518298826*10, -3.18755225906466*10],
            [ 2.40085652217597,  -2.63125454261783,  1.54912674773593,  -3.31172956504304*10**(-1)]]
        param_dict["coef"] = torch.tensor(coeflist)
        param_dict["degree"] = [4,4]
        param_dict["scale"] = 0
        self.param_nest_dict[param_dict["name"]] =  param_dict



        #f2g3
        param_dict= {"name":str, "coef": torch.Tensor, "degree": list, "scale" : Union[int, torch.Tensor]}
        param_dict["name"] = "f2g3"
        coeflist = [[1.875, -1.25,  0.375,0],
            [4.4814453125, -16.1884765625,  25.013671875, -12.55859375]]
        param_dict["coef"] = torch.tensor(coeflist)
        param_dict["degree"] = [3,4]
        param_dict["scale"] = 0
        self.param_nest_dict[param_dict["name"]] =  copy.deepcopy(param_dict)

        #2f12g1
        param_dict= {"name":str, "coef": torch.Tensor, "degree": list, "scale" : Union[int, torch.Tensor]}
        param_dict["name"] = "2f12g1"
        coeflist = [[1.5, -0.5],
            [1.5, -0.5],
            [2.076171875,  -1.3271484375],
            [2.076171875,  -1.3271484375]]
        param_dict["coef"] = torch.tensor(coeflist)
        param_dict["degree"] = [2,2,2,2]
        param_dict["scale"] = 0
        self.param_nest_dict[param_dict["name"]] =  copy.deepcopy(param_dict)

        #f2g2
        param_dict= {"name":str, "coef": torch.Tensor, "degree": list, "scale" : Union[int, torch.Tensor]}
        param_dict["name"] = "f2g2"
        coeflist = [[1.875, -1.25,  0.375],
            [3.255859375,  -5.96484375,  3.70703125]]
        param_dict["coef"] = torch.tensor(coeflist)
        param_dict["degree"] = [3,3]
        param_dict["scale"] = 0
        self.param_nest_dict[param_dict["name"]] =  copy.deepcopy(param_dict)


        #f1g2
        param_dict= {"name":str, "coef": torch.Tensor, "degree": list, "scale" : Union[int, torch.Tensor]}
        param_dict["name"] = "f1g2"
        coeflist = [[1.5, -0.5,  0],
            [3.255859375,  -5.96484375,  3.70703125]]
        param_dict["coef"] = torch.tensor(coeflist)
        param_dict["degree"] = [2,3]
        param_dict["scale"] = 0
        self.param_nest_dict[param_dict["name"]] =  copy.deepcopy(param_dict)
        


def run_set(net, data_loader, device='cuda:0'):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = net(images)

    print(images.shape)


def download_url(url, model_dir , overwrite=False):
	
	try:
		
		if not os.path.exists(cached_file) or overwrite:
			sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
			urlretrieve(url, cached_file)
		return cached_file
	except Exception as e:
		# remove lock file so download can be executed next time.
		# os.remove(os.path.join(model_dir, 'download.lock'))
		# sys.stderr.write('Failed to download from url %s' % url + '\n' + str(e) + '\n')
		return None


if __name__ == "__main__":
    global_config = load_model_yaml("./global_config/", "global_config.yaml")
    parser = ArgumentParser()
    parser.add_argument("-dd", "--download_dataset", type=bool, choices=[True, False])
    parser.add_argument("--dataset", type=str,choices=["cifar10", "imagenet_1k", "cifar100"])
    args = parser.parse_args()
    print(args)
    if(args.download_dataset):
        os.makedirs(global_config["Global"]["dataset_dirctory"], exist_ok=True)
        if(args.dataset == "cifar10"):
            datasets.CIFAR10(root = global_config["Global"]["dataset_dirctory"], download=True)
        elif(args.dataset == "cifar100"):
            datasets.CIFAR100(root = global_config["Global"]["dataset_dirctory"], download=True)
        elif(args.dataset == "imagenet_1k"):
            url = 'https://hanlab.mit.edu/files/OnceForAll/ofa_cvpr_tutorial/imagenet_1k.zip'
            target_dir = url.split('/')[-1]
            model_dir = os.path.expanduser(global_config["Global"]["dataset_dirctory"])
            dataset_dir = model_dir
            model_dir = os.path.join(model_dir, args.dataset)
            if not os.path.exists(model_dir):
                 os.makedirs(model_dir)
            cached_file = model_dir
            os.system(f"wget {url} --no-check-certificate")
            os.system(f"mv imagenet_1k.zip {model_dir}")
            os.system(f"unzip {model_dir}/imagenet_1k.zip -d {dataset_dir}")
        else:
             raise Exception("dataset error")

           