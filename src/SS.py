from util import *
from custom_module import *
from pretrained_model import *
global_config = load_model_yaml("./global_config/", "global_config.yaml")

def generate_layer_input_scale(model: nn.Module, train_data_loader, layer_nest_dict, directory_path):
    if(not os.path.exists(directory_path)):
            os.mkdir(directory_path)
    data_type = "_scale"
    for key in layer_nest_dict:
        my_model = model
        layer_name = key
        print("name: " + layer_name)
        collection_layer = Input_scale_collection_layer(layer_name, access_layer(my_model, layer_name))
        replace_layer(my_model, layer_name, collection_layer)
        print()

    run_set(my_model, train_data_loader, "cuda:0")

    for key in layer_nest_dict:
        layer_name = key
        access_layer(my_model, layer_name).save(directory_path, layer_name + data_type + ".pt")
        data = torch.load(directory_path + layer_name + data_type + ".pt")
        print(data)


def CT_reset_scale(model, sign_scale, scale_path, scale_ratio, sign_nest_dict):
    model = model.to("cuda:0")
    for key in sign_nest_dict:
        scale_name = key + "_scale.pt"
        if(scale_path != None):
            sign_scale = torch.load(scale_path + scale_name).item()
            print("scale: " + str(sign_scale))
        access_layer(model, key).sign.scale = sign_scale
        access_layer(model, key).sign.scale_ratio = scale_ratio


def SS_replace(model,valid_data_loader, train_data_loader, sign_type, input_data_dirctory):
    model = model
    sign_nest_dict = generate_sign_nest_dict(model)
    dirctory = input_data_dirctory + "model_PR_AT/"
    file_name = "model_PR_AT_"+sign_type+".pt"
    scale_path = input_data_dirctory + "Scale_" + sign_type + "/"
    model = torch.load(dirctory+file_name)
    validate(model, valid_data_loader, "cuda:0")
    generate_layer_input_scale(model = copy.deepcopy(model), train_data_loader = train_data_loader, layer_nest_dict = sign_nest_dict, directory_path = scale_path)
    CT_reset_scale(model = model, sign_scale = 100, scale_path= scale_path, scale_ratio = 1, sign_nest_dict= sign_nest_dict)
    validate(model, valid_data_loader, "cuda:0")
    file_name2 = "model_PR_AT_SS_"+sign_type+".pt"
    torch.save(model, dirctory+file_name2)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str,choices=["vgg19_bn", "resnet18", "resnet32"])
    parser.add_argument("--dataset", type=str,choices=["cifar10", "imagenet_1k", "cifar100"])
    parser.add_argument("-wd", "--working_directory", type=str, default="./working_dirctory/")
    parser.add_argument("-st","--sign_type", type=str, default="a7", choices=["a7", "2f12g1", "f1g2", "f2g2", "f2g3"])
    args = parser.parse_args()
    print(args)

    valid_data_loader = None
    train_data_loader = None
    valid_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "valid", data_dir = global_config["Global"]["dataset_dirctory"])
    train_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "train", data_dir = global_config["Global"]["dataset_dirctory"])


    # if(args.dataset == "cifar10" or args.dataset == "cifar100"):
    #     valid_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "valid", data_dir = global_config["Global"]["dataset_dirctory"])
    #     train_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "train", data_dir = global_config["Global"]["dataset_dirctory"])
    # elif(args.dataset == "imagenet_1k"):
    #     valid_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "valid", data_dir = os.path.join(global_config["Global"]["dataset_dirctory"], args.dataset) )
    #     train_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "train", data_dir = os.path.join(global_config["Global"]["dataset_dirctory"], args.dataset) )
   
    model = get_pretrained_model(model_name=args.model, dataset=args.dataset)
    
    SS_replace(model = model, valid_data_loader=valid_data_loader, train_data_loader=train_data_loader ,sign_type = args.sign_type, input_data_dirctory = args.working_directory)
