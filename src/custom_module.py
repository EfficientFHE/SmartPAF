import os
import math
from typing import Tuple, Union
import torch 
import torch.nn as nn
import copy
from math import pi, sqrt



# TODO: PA+CT instead of CT -> PA.
class Input_data_collection_layer(nn.Module):
    def __init__(self, name, layer : nn.Module):
        super().__init__()
        self.data_store = torch.tensor([])
        self.name = name
        self.layer = layer

        
    def forward(self, x):
        x_backup = x[None, :].to("cpu")
        self.data_store = torch.cat((self.data_store, x_backup), 0)

        res =self.layer.forward(x)
        return res


    def save(self, directory , file_name):
        if(not os.path.exists(directory)):
            os.mkdir(directory)
        torch.save(self.data_store, directory + file_name)



class Input_scale_collection_layer(nn.Module):
    def __init__(self, name, layer : nn.Module):
        super().__init__()
        self.have_first = False
        self.data_store = torch.tensor(0)
        self.name = name
        self.layer = layer

        
    def forward(self, x):
        s_max = torch.max(x).item()
        s_min = torch.min(x).item()
        scale = max(abs(s_max), abs(s_min))
        if(scale > self.data_store.item()):
            self.data_store = torch.tensor(scale)
        res =self.layer.forward(x)
        return res


    def save(self, directory , file_name):
        if(not os.path.exists(directory)):
            os.mkdir(directory)
        torch.save(self.data_store, directory + file_name)


        
class Sign_minmax_layer(nn.Module):
    def __init__(self, coef, degree, scale = 0, scale_ratio = 1, train_coef = True, param_scale = False):
        super().__init__()
        self.scale_ratio = scale_ratio
        self.degree = degree
        self.coeflist = nn.Parameter(coef.to("cuda:0"), requires_grad=train_coef).to("cuda:0")
        self.param_scale = param_scale
        if(self.param_scale):
            self.scale =  nn.Parameter(torch.tensor(scale).to("cuda:0"), requires_grad=False).to("cuda:0")
        else:
            self.scale = scale
            

    def forward(self, x):

        if(self.scale == 0):
            s_max = torch.max(x).item()
            s_min = torch.min(x).item()
            scale = max(abs(s_max), abs(s_min)) * self.scale_ratio
        else:
            if(self.param_scale):
                scale = self.scale.item() * self.scale_ratio
            else:
                scale = self.scale * self.scale_ratio


        # x_bk = torch.clone(x).to(x.device)
        x = torch.divide(x, scale)

        coeflist = self.coeflist

        for compositive_id in range(coeflist.shape[0]):

            degree_num = self.degree[compositive_id]

            # x_degree_1 = torch.clone(x).to(x_bk.device)
            x_degree_2 = torch.mul(x, x)

            # out = torch.clone(x).to(x_bk.device)
            out = torch.mul(x, coeflist[compositive_id][0]) # x^1 * coe[1]

            for i in range(1, degree_num):
                x = torch.mul(x, x_degree_2)
                partial_out = torch.mul(x, coeflist[compositive_id][i])
                out = torch.add(out, partial_out)
            x = torch.clone(out).to(x.device)
        

        result = out.to(x.device)
        del x

        return result
    
    def set_coef_grad(self, grad):
        self.coeflist.requires_grad = grad

    def set_scale_grad(self, grad):
        if(self.param_scale):
            self.scale.requires_grad = grad

    def save_coef(self, path_name):
        torch.save(self.coeflist, path_name)

    def save_scale(self, path_name):
        if(self.param_scale):
            torch.save(self.scale, path_name)


class Sigmoid_minmax_layer(nn.Module):
    def __init__(self, coef, degree, scale = 0, scale_ratio = 1, train_coef = True, param_scale = False):
        super().__init__()
        self.scale_ratio = scale_ratio
        self.degree = degree
        self.coeflist = nn.Parameter(coef.to("cuda:0"), requires_grad=train_coef).to("cuda:0")
        self.param_scale = param_scale
        if(self.param_scale):
            self.scale =  nn.Parameter(torch.tensor(scale).to("cuda:0"), requires_grad=False).to("cuda:0")
        else:
            self.scale = scale
            

    def forward(self, x):

        if(self.scale == 0):
            s_max = torch.max(x).item()
            s_min = torch.min(x).item()
            scale = max(abs(s_max), abs(s_min)) * self.scale_ratio
        else:
            if(self.param_scale):
                scale = self.scale.item() * self.scale_ratio
            else:
                scale = self.scale * self.scale_ratio


        # x_bk = torch.clone(x).to(x.device)
        x = torch.divide(x, scale)

        coeflist = self.coeflist

        for compositive_id in range(coeflist.shape[0]):

            degree_num = self.degree[compositive_id]

            # x_degree_1 = torch.clone(x).to(x_bk.device)
            x_degree_2 = torch.mul(x, x)

            # out = torch.clone(x).to(x_bk.device)
            out = torch.mul(x, coeflist[compositive_id][0]) # x^1 * coe[1]

            for i in range(1, degree_num):
                x = torch.mul(x, x_degree_2)
                partial_out = torch.mul(x, coeflist[compositive_id][i])
                out = torch.add(out, partial_out)
            x = torch.clone(out).to(x.device)
        
        x = x * 0.5 + 0.5
        result = out.to(x.device)
        del x

        return result
    
    def set_coef_grad(self, grad):
        self.coeflist.requires_grad = grad

    def set_scale_grad(self, grad):
        if(self.param_scale):
            self.scale.requires_grad = grad

    def save_coef(self, path_name):
        torch.save(self.coeflist, path_name)

    def save_scale(self, path_name):
        if(self.param_scale):
            torch.save(self.scale, path_name)




class ReLU_sign_layer(nn.Module):
    def __init__(self, sign:nn.Module):
        super().__init__()
        self.sign = sign

    def forward(self, x):
        result = torch.divide(torch.add(x, torch.mul(x, self.sign.forward(x))),2)
        return result
    
class SiLU_minmax_layer(nn.Module):
    def __init__(self, sigmoid:nn.Module):
        super().__init__()
        self.sigmoid = sigmoid

    def forward(self, x):
        result = torch.mul(x, self.sign.forward(x))
        return result
    


class Maxpool_sign_layer(nn.Module):
    def __init__(self, sign:nn.Module, 
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride : Union[int, Tuple[int, int]] = 0, 
                 padding : Union[int, Tuple[int, int]] = 0, 
                 dilation : Union[int, Tuple[int, int]] = 1, ):
        super().__init__()

        self.sign = sign

        self.kernel_size = self.to_tuple(kernel_size)

        if(stride == 0):
            self.stride = self.kernel_size
        else:
            self.stride = self.to_tuple(stride)
        self.padding = self.to_tuple(padding)
        self.dilation = self.to_tuple(dilation)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

        

    def forward(self, x): 
        x_size = x.size() 
        x_unfold = self.unfold(x)
        x_unfold_size = x_unfold.size()
        x_reshape = torch.reshape(x_unfold, (x_unfold_size[0], -1, int(x_unfold_size[1] / x_size[1]), x_unfold_size[2]))

        h_out = math.floor((x_size[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] -1) - 1)/ self.stride[0] + 1)
        w_out = math.floor((x_size[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] -1) - 1)/ self.stride[1] + 1)
        result = self.maxpool(x_reshape).reshape((x_size[0], x_size[1], h_out, w_out))
        return result
    

    def to_tuple(self, param:Union[int, Tuple[int, int]]):
        if(isinstance(param, int)):
            param = (param, param)
        return param
        
    def maxpool(self, x):
        x_size = x.size()
        pool_size = x_size[2]

        if(pool_size == 1):
            return x
        elif(pool_size == 2):
            return self.max(x[:,:,0,:], x[:,:,1,:])
        else:
            pivot = int(pool_size / 2)
            a = self.maxpool(x[:,:,0:pivot,:])
            b = self.maxpool(x[:,:,pivot:pool_size,:])
            return self.max(a, b)


    def max(self, a, b):
        a = torch.squeeze(a)
        b = torch.squeeze(b)
        sum = torch.add(a,b)
        diff = torch.sub(a,b)
        sign_diff = self.sign.forward(diff)
        result = torch.divide(torch.add(sum, torch.mul(sign_diff, diff)), 2)
        return result
    

class HerPN2d(nn.Module):
    @staticmethod
    def h0(x):
        return torch.ones(x.shape).to(x.device)

    @staticmethod
    def h1(x):
        return x

    @staticmethod
    def h2(x):
        return (x * x - 1)

    def __init__(self, num_features : int, BN_dimension=2 ,BN_copy:nn.Module = None):
        super().__init__()
        self.f = (1 / sqrt(2 * pi), 1 / 2, 1 / sqrt(4 * pi))
        
        if(BN_copy):
            self.bn0 = copy.deepcopy(BN_copy)
            self.bn1 = copy.deepcopy(BN_copy)
            self.bn2 = copy.deepcopy(BN_copy)

        elif(BN_dimension == 1):
            self.bn0 = nn.BatchNorm1d(num_features)
            self.bn1 = nn.BatchNorm1d(num_features)
            self.bn2 = nn.BatchNorm1d(num_features)
        else:
            self.bn0 = nn.BatchNorm2d(num_features)
            self.bn1 = nn.BatchNorm2d(num_features)
            self.bn2 = nn.BatchNorm2d(num_features)



        self.bn = (self.bn0, self.bn1, self.bn2)
        self.h = (self.h0, self.h1, self.h2)


    def forward(self, x):
        result = torch.zeros(x.shape).to(x.device)
        for bn, f, h in zip(self.bn, self.f, self.h):
            poly = torch.mul(f, h(x))
            # print(poly.shape)
            result = torch.add(result, bn(poly))

        return result
    

class Sigmoid_minmax_layer(nn.Module):
    def __init__(self, coef, degree, scale = 0, scale_ratio = 1, train_coef = True, param_scale = False):
        super().__init__()
        self.scale_ratio = scale_ratio
        self.degree = degree
        self.coeflist = nn.Parameter(coef.to("cuda:0"), requires_grad=train_coef).to("cuda:0")
        self.param_scale = param_scale
        if(self.param_scale):
            self.scale =  nn.Parameter(torch.tensor(scale).to("cuda:0"), requires_grad=False).to("cuda:0")
        else:
            self.scale = scale
            

    def forward(self, x):

        if(self.scale == 0):
            s_max = torch.max(x).item()
            s_min = torch.min(x).item()
            scale = max(abs(s_max), abs(s_min)) * self.scale_ratio
        else:
            if(self.param_scale):
                scale = self.scale.item() * self.scale_ratio
            else:
                scale = self.scale * self.scale_ratio


        # x_bk = torch.clone(x).to(x.device)
        x = torch.divide(x, scale)

        coeflist = self.coeflist

        for compositive_id in range(coeflist.shape[0]):

            degree_num = self.degree[compositive_id]

            # x_degree_1 = torch.clone(x).to(x_bk.device)
            x_degree_2 = torch.mul(x, x)

            # out = torch.clone(x).to(x_bk.device)
            out = torch.mul(x, coeflist[compositive_id][0]) # x^1 * coe[1]

            for i in range(1, degree_num):
                x = torch.mul(x, x_degree_2)
                partial_out = torch.mul(x, coeflist[compositive_id][i])
                out = torch.add(out, partial_out)
            x = torch.clone(out).to(x.device)
        
        result = (out + 0.5).to(x.device)
        del x

        return result
    
    def set_coef_grad(self, grad):
        self.coeflist.requires_grad = grad

    def set_scale_grad(self, grad):
        if(self.param_scale):
            self.scale.requires_grad = grad

    def save_coef(self, path_name):
        torch.save(self.coeflist, path_name)

    def save_scale(self, path_name):
        if(self.param_scale):
            torch.save(self.scale, path_name)

class SiLU_minmax_layer(nn.Module):
    def __init__(self, sigmoid:nn.Module):
        super().__init__()
        self.sigmoid = sigmoid

    def forward(self, x ):


        result = torch.mul(x, self.sigmoid.forward(x))
        return result
    
class SiLU_minmax_bn_layer(nn.Module):
    def __init__(self, sigmoid:nn.Module, num_features):
        super().__init__()
        self.sigmoid = sigmoid
        self.bn = nn.BatchNorm2d(num_features).to("cuda:0")

    def forward(self, x ):


        result = torch.mul(x, self.sigmoid.forward(x))
        result = self.bn(result)
        return result