from util import *
from custom_module import *
from pretrained_model import *
global_config = load_model_yaml("./global_config/", "global_config.yaml")

def generate_layer_input_data(model: nn.Module , layer_nest_dict, directory_path,  train_data_loader):
    if(not os.path.exists(directory_path)):
            os.mkdir(directory_path)
    data_type = "_input"
    for key in layer_nest_dict:
        my_model = copy.deepcopy(model)
        layer_name = key
        print("name: " + layer_name)
        collection_layer = Input_data_collection_layer(layer_name, access_layer(my_model, layer_name))
        replace_layer(my_model, layer_name, collection_layer)
        run_set(my_model, train_data_loader, "cuda:0")
        access_layer(my_model, layer_name).save(directory_path, layer_name + data_type + ".pt")
        data = torch.load(directory_path + layer_name + data_type + ".pt")
        print(data.shape)


def generate_data_set(dirctory_path , layer_nest_dict, split_point):
    train_path = "train/"
    valid_path = "val/"
    if(not os.path.exists(dirctory_path + train_path)):
        os.mkdir(dirctory_path + train_path)
    if(not os.path.exists(dirctory_path + valid_path)):
        os.mkdir(dirctory_path + valid_path)   

    for key in layer_nest_dict:
        data_type = "_input"
        layer_name = key
        file_name = layer_name + data_type + ".pt"
        print(layer_name)
        data = torch.load(dirctory_path + file_name)
        data = data.reshape((-1, ) + data.shape[2:])
        b=torch.randperm(data.shape[0])
        data = data[b]
        train_data = data[0:split_point]
        valid_data = data[split_point:data.shape[0]]
        torch.save(train_data, dirctory_path + train_path + file_name)
        torch.save(valid_data, dirctory_path + valid_path + file_name)
        print(train_data.shape)
        print(valid_data.shape)


def data_collection(model, valid_data_loader, train_data_loader, split_point, input_data_save_path):
    sign_nest_dict = generate_sign_nest_dict(model)
    validate(model, valid_data_loader)
    generate_layer_input_data(model, sign_nest_dict, input_data_save_path, train_data_loader)
    generate_data_set(input_data_save_path , sign_nest_dict, split_point)


def CT_train(sign_type, sign_scale, scale_path, sign_nest_dict,batch_size, input_data_dirctory, output_floder_suffix):
    sign_param_dict = Sign_parameter_generator().param_nest_dict[sign_type]
    print(sign_type)
    for key in sign_nest_dict:
        sign_dict = sign_nest_dict[key]
        scale_name = key + "_scale.pt"
        if(scale_path != None):
            sign_scale = torch.load(scale_path + scale_name).item()
            print("scale: " + str(sign_scale))
        sign_module = Sign_minmax_layer(coef=sign_param_dict["coef"], degree=sign_param_dict["degree"],scale=sign_scale)
        print("name: ", key)
        if(sign_dict["type"] == "ReLU"):
            my_model = ReLU_sign_layer(sign = sign_module)
            ref_model = nn.ReLU()
        elif(sign_dict["type"] == "MaxPool2d"):
            my_model = Maxpool_sign_layer(sign = sign_module, kernel_size=sign_dict["kernel_size"], stride= sign_dict["stride"], padding=sign_dict["padding"], dilation=sign_dict["dilation"])
            ref_model = nn.MaxPool2d(kernel_size=sign_dict["kernel_size"], stride= sign_dict["stride"], padding=sign_dict["padding"], dilation=sign_dict["dilation"])
        else:
            raise Exception("not implemented layer type.")
        optimizer = torch.optim.Adam(params=my_model.parameters(), lr=0.01, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 2, threshold= 1e-8, min_lr= 1e-4)
        train_path = "train/"
        val_path = "val/"
        data_type = "_input"
        file_name = key + data_type + ".pt"
        print(file_name)
        train_data = torch.load(input_data_dirctory + train_path + file_name)
        valid_data = torch.load(input_data_dirctory + val_path + file_name)
        for epoch_i in range(40):
            train_loss_meter = AverageMeter("train loss")
            val_loss_meter = AverageMeter("val loss")   
            #train
            for batch_i in range(int(train_data.shape[0] / batch_size)):
                x = train_data[batch_i * batch_size : (batch_i + 1) * batch_size].to("cuda:0")
                target_y = ref_model.to("cuda:0").forward(x)
                actual_y = my_model.forward(x)
                loss_fun = nn.MSELoss()
                my_model.zero_grad()
                loss = loss_fun(actual_y, target_y)
                train_loss_meter.update(val=float(loss.cpu().item()), n=x.shape[0])
                loss.backward()
                optimizer.step()
            train_loss = train_loss_meter.avg

            #valid
            for batch_i in range(int(valid_data.shape[0] / batch_size)):
                x = valid_data[batch_i * batch_size : (batch_i + 1) * batch_size].to("cuda:0")
                target_y = ref_model.to("cuda:0").forward(x)
                actual_y = my_model.forward(x)
                loss_fun = nn.MSELoss()
                loss = loss_fun(actual_y, target_y)
                val_loss_meter.update(val=float(loss.cpu().item()), n=x.shape[0])
            val_loss = val_loss_meter.avg
        
            scheduler.step(val_loss)

            print(
                f"Epoch:{epoch_i + 1}"
                + f" Train Loss:{train_loss:.10f}"
                + f" Val Loss: {val_loss:.10f}"
            )

        folder_name = "CT_" + sign_type + "_S" + output_floder_suffix+"_40s/"
        coef_save_dirctory = input_data_dirctory + folder_name
        if(not os.path.exists(coef_save_dirctory)):
                os.mkdir(coef_save_dirctory)
        file_name = key + "_coef.pt"
        my_model.sign.save_coef(coef_save_dirctory + file_name)
        print("save: " + folder_name + file_name)
        print("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str,choices=["vgg19_bn", "resnet18", "resnet32"])
    parser.add_argument("--dataset", type=str,choices=["cifar10", "cifar100", "imagenet_1k"])
    parser.add_argument("-st","--sign_type", type=str, choices=["a7", "2f12g1", "f1g2", "f2g2", "f2g3"])
    parser.add_argument("-dc","--data_collection", type=bool, default=False, choices=[True , False])
    parser.add_argument("-wd", "--working_directory", type=str, default="./working_directory/")

    args = parser.parse_args()
    print(args)
    if(args.dataset == "cifar10" or args.dataset == "cifar100"):
        split_point = 45000
        batch_size = 100
    elif(args.dataset == "imagenet_1k"):
        split_point = 900
        batch_size = 40
    model = get_pretrained_model(model_name=args.model, dataset=args.dataset)
    if(args.data_collection):
        data_collection(model = model,
            valid_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "valid", data_dir = global_config["Global"]["dataset_dirctory"] ),
            train_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "train", data_dir = global_config["Global"]["dataset_dirctory"] ),
            split_point = split_point, input_data_save_path = args.working_directory)
        
        '''
        if(args.dataset == "cifar10" or args.dataset == "cifar100"):
            data_collection(model = model,
                        valid_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "valid", data_dir = global_config["Global"]["dataset_dirctory"] ),
                        train_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "train", data_dir = global_config["Global"]["dataset_dirctory"] ),
                        split_point = split_point, input_data_save_path = args.working_directory)
        elif(args.dataset == "imagenet_1k"):
            data_collection(model = model,
                        valid_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "valid", data_dir = os.path.join(global_config["Global"]["dataset_dirctory"], args.dataset) ),
                        train_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "train", data_dir = os.path.join(global_config["Global"]["dataset_dirctory"], args.dataset) ),
                        split_point = split_point, input_data_save_path = args.working_directory)
        '''
        
    else:
        nest_dict = generate_sign_nest_dict(model) 
        CT_train(sign_type = args.sign_type, sign_scale = 0, scale_path= None, sign_nest_dict = nest_dict,batch_size = batch_size,
                 input_data_dirctory = args.working_directory , output_floder_suffix= "dynamic")
