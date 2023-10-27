from util import *
from custom_module import *
from pretrained_model import *
torch.manual_seed(0)
global_config = load_model_yaml("./global_config/", "global_config.yaml")

global G_model
global R_model
global B_model
global E_model
global S_model


def print_log_to_file(log_file, config, train_log, type, before_acc, swa_log):
    with open(log_file, "a") as f:
        print(" \n", file=f)
        print("config: ", file = f)
        print("before_acc: " + str(before_acc), file=f)
        print(config, file= f)
        if(train_log):
            acc_log_list = train_log["train_result"]["va"]
            if(type == "s"):
                end_i = len(acc_log_list)
            elif(type == "b"):
                end_i = train_log["best_index"] + 1
            for i in range(end_i):
                print("acc: " + str(acc_log_list[i]), file = f)
            print("swa: "+str(swa_log), file= f)


def get_optimizer(
    model: torch.nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Returns the optimizer initializer according to the config

    Note: config has a minimum of three entries.
    Feel free to add more entries if you want.
    But do not change the name of the three existing entries

    Args:
    - model: the model to optimize for
    - config: a dictionary containing parameters for the config
    Returns:
    - optimizer: the optimizer
    """

    optimizer = None

    optimizer_type = config.get("optimizer_type", "sgd")
    learning_rate = config.get("lr", 0)
    weight_decay = config.get("weight_decay", 0)
    momentum = 0
    dampening = 0

    print(learning_rate)
    print(weight_decay)
    print(optimizer_type)
    if optimizer_type=="sgd":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum = momentum, dampening = dampening)
    elif optimizer_type=="adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy given the prediction logits and the ground-truth labels

    Args:
        logits: The output of the forward pass through the model.
                for K classes logits[k] (where 0 <= k < K) corresponds to the
                log-odds of class `k` being the correct one.
                Shape: (batch_size, num_classes)
        labels: The ground truth label for each instance in the batch
                Shape: (batch_size)
    Returns:
        accuracy: The accuracy of the predicted logits
                   (number of correct predictions / total number of examples)
    """
    batch_accuracy = 0.0
    num_data = logits.size()[0]
    for _i in range(num_data):
        nn_inference_label = torch.argmax(logits[_i])
        if(labels[_i] == nn_inference_label):
            batch_accuracy += 1.0

    batch_accuracy = batch_accuracy / num_data

    return batch_accuracy


def compute_loss(
    model: nn.Module,
    model_output: torch.Tensor,
    target_labels: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    """
    Computes the loss between the model output and the target labels

    Args:
    -   model: a model (which inherits from nn.Module)
    -   model_output: the raw scores output by the net
    -   target_labels: the ground truth class labels
    -   is_normalize: bool flag indicating that loss should be divided by the batch size
    Returns:
    -   the loss value
    """
    loss = None

    criterion = nn.CrossEntropyLoss()
    loss = criterion(model_output, target_labels)
    #loss = model.loss_criterion(model_output, target_labels)
    
    if(is_normalize):
        loss = loss / model_output.size()[0]

    return loss


class Trainer:
    """Class that stores model training metadata."""
    def __init__(
        self,
        #data_dir: str,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model_dir: str = "None",
        load_from_disk: bool = True,
        cuda: bool = True,
        lr_scheduler: bool = False,
        no_bn_track: bool = True,
    ) -> None:

        self.model_dir = model_dir
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.no_bn_track = no_bn_track

        self.cuda = cuda
        if cuda:
            self.model.cuda()

        self.train_loader = train_loader                   
        self.val_loader = val_loader

        self.optimizer = optimizer

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

        # load the model from the disk if it exists
        if os.path.exists(model_dir) and load_from_disk:
            checkpoint = torch.load(os.path.join(self.model_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.train()


    def save_model(self) -> None:
        """
        Saves the model state and optimizer state on the dict
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.model_dir, "checkpoint.pt"),
        )


    def run_training_loop(self, num_epochs: int, swa_pack = None) -> None:
        """Train for num_epochs, and validate after every epoch."""

        best_val = 0.0
        best_loss = 100
        best_epoch_i = 0
        train_result = {"tl": [], "vl": [], "ta" :[], "va":[]}

        for epoch_idx in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_acc)
            val_loss, val_acc = self.validate()
            self.validation_loss_history.append(val_loss)
            self.validation_accuracy_history.append(val_acc)

            if(swa_pack != None and num_epochs > swa_pack[2]):
                swa_pack[0].update_parameters(self.model)
                swa_pack[1].step()

            train_result["tl"].append(train_loss)
            train_result["vl"].append(val_loss)
            train_result["ta"].append(train_acc)
            train_result["va"].append(val_acc)
            
            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Val Loss: {val_loss:.4f}"
                + f" Train Accuracy: {train_acc:.4f}"
                + f" Validation Accuracy: {val_acc:.4f}"
            )

            global B_model
            if(val_acc > best_val):
                best_val = val_acc
                best_loss = val_loss
                B_model = copy.deepcopy(self.model)
                best_epoch_i = epoch_idx
            elif(val_acc == best_val and val_loss < best_loss):
                best_val = val_acc
                best_loss = val_loss
                B_model = copy.deepcopy(self.model)
                best_epoch_i = epoch_idx
        if(swa_pack != None):
            torch.optim.swa_utils.update_bn( self.train_loader,swa_pack[0].to("cpu"))

        return_pack={"train_result" : train_result, "best_index": best_epoch_i}
        return return_pack


    def train_epoch(self) -> Tuple[float, float]:
        """Implements the main training loop."""
        self.model.train()

        if(self.no_bn_track):
            self.disable_traking_bn()

        train_loss_meter = AverageMeter("train loss")
        train_acc_meter = AverageMeter("train accuracy")

        # loop over each minibatch
        for (x, y) in self.train_loader:
            if self.cuda:
                x = x.cuda()
                y = y.cuda()

            n = x.shape[0]
            logits = self.model(x)
            batch_acc = compute_accuracy(logits, y)
            train_acc_meter.update(val=batch_acc, n=n)

            batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
            train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        return train_loss_meter.avg, train_acc_meter.avg


    def validate(self) -> Tuple[float, float]:
        """Evaluate on held-out split (either val or test)"""
        self.model.eval()

        val_loss_meter = AverageMeter("val loss")
        val_acc_meter = AverageMeter("val accuracy")

        # loop over whole val set
        with torch.no_grad():
            for (x, y) in self.val_loader:
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()

                n = x.shape[0]
                logits = self.model(x)

                batch_acc = compute_accuracy(logits, y)
                val_acc_meter.update(val=batch_acc, n=n)

                batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
                val_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

        return val_loss_meter.avg, val_acc_meter.avg


    def disable_traking_bn(self):
        for layer in self.model.modules():
            if isinstance(layer, nn.modules.BatchNorm2d):
                layer.eval()


def train_group(model, valid_data_loader, train_data_loader, config, relu_list):

    layer_name = config["layer_name"]
    num_epochs = config["ep"]

    optimizer_name = "adam"
    learning_rate = config["lr"]
    weight_decay = config["wd"]
    learning_rate_decay = False
    no_bn_track = True

    train_weight = config["tw"]
    train_weight_exception = config["twe"]
    train_coef = config["tc"]
    train_coef_exception = config["tce"]

    scale_ratio = 1
    augment = False
    dropout = config["do"]
    swa = True
    swa_start = 0

    need_replace = False
    train_coef_replace = False
    dynamic_scale = True

    my_model = model
    relu_list = relu_list

    for i in my_model.parameters():
        i.requires_grad = train_weight

    for ex_name in train_weight_exception:
        for i in access_layer(my_model, ex_name).parameters():
            i.requires_grad = not train_weight

    for relu_name in relu_list:
        for i in access_layer(my_model, relu_name).parameters():
            i.requires_grad = train_coef

    if(need_replace and train_coef_replace != train_coef):
        train_coef_exception.append(layer_name)
        
    for ex_relu_name in train_coef_exception:
        for i in access_layer(my_model, ex_relu_name).parameters():
            i.requires_grad = not train_coef

    if(global_config["PA_AT"]["dropout_enable"]):
        if(dropout):
            if((not isinstance(my_model.fc, nn.Sequential))):
                my_model.fc = nn.Sequential(nn.Dropout(p = 0.5), my_model.fc)
        elif(isinstance(my_model.fc, nn.Sequential)):
            my_model.fc = my_model.fc[1]

    print("Name: " + layer_name)
    before_result  = validate(my_model, valid_data_loader, "cuda:0")

    print(layer_name +": train")

    optimizer_config = {"optimizer_type": "adam", "lr": learning_rate, "weight_decay": weight_decay}
    optimizer = get_optimizer(my_model, optimizer_config)
    trainer = Trainer(
        model=my_model,
        optimizer=optimizer,
        load_from_disk=False,
        cuda=True,
        lr_scheduler = learning_rate_decay,
        no_bn_track = no_bn_track,
        train_loader = train_data_loader,
        val_loader = valid_data_loader
    )

    if(swa):
        swa_model = AveragedModel(my_model)
        swa_scheduler = SWALR(optimizer, swa_lr=learning_rate)
        swa_pack = [swa_model, swa_scheduler, swa_start]
    else:
        swa_pack = None

    print("Layer name: " + layer_name)
    print("Parameter: ")
    print("Optimizer: " + optimizer_name)
    print("No batchnorm tracking: " +str(no_bn_track))
    print("\tLearning rate: " + str(learning_rate))
    print("\tWeight decay: " + str(weight_decay))
    print("\tLearning rate decay: " + str(learning_rate_decay))
    print("Train weight: " + str(train_weight))
    print("\tException:" + str(train_weight_exception))
    print("Train coefficient: " + str(train_coef))
    print("\tException: " + str(train_coef_exception))
    print("\tReplace: " + str(need_replace))
    print("\t\tDynamic scale: " + str(dynamic_scale))
    print("Scale ratio: " + str(scale_ratio))
    print("Augment: " + str(augment))
    print("Dropout: " + str(dropout))
    print("SWA: " + str(swa))
    print("\tStart: "+ str(swa_start))
    print("\n \n")

    print("Train epoch: ")
    train_return_pack = trainer.run_training_loop(num_epochs=num_epochs, swa_pack=swa_pack)
    train_result = validate(my_model, valid_data_loader, "cuda:0")
    global E_model
    E_model = copy.deepcopy(my_model)
    if(swa):
        swa_result = validate(swa_model,valid_data_loader, "cuda:0")
        global S_model
        S_model = copy.deepcopy(swa_model.module)
        swa_acc = swa_result[1] / 100.0
    best_acc = train_return_pack["train_result"]["va"][train_return_pack["best_index"]]

    acc_delta = [x - y for x, y in zip(train_return_pack["train_result"]["ta"],train_return_pack["train_result"]["va"])]
    if(max(acc_delta) >= 0.1):
        overfit = True
    else:
        overfit = False

    print("\n \n")
    print("Validation result:")
    print("\tbefore: " + str(before_result))
    print("\ttrain: "+ str(train_result))
    if(swa):
        print("\tswa: "+ str(swa_result))

    print("\n \n \n \n")

    return_pack = {"best_acc": best_acc, "swa_acc": swa_acc, "overfit" : overfit, "hash" : 0, "train_log" : train_return_pack}
    return return_pack


def train_layer(my_model = None, layer_name = None, sign_nest_dict = None, valid_data_loader = None,train_data_loader = None,  group_epochs = 20, replace_module = None, lr_c = None, lr_w = None, log_file = None):
    global G_model
    global S_model
    global B_model
    global R_model

    relu_list = list(sign_nest_dict.keys())
    sign_layer_list = []
    sign_layer_list.append(sign_nest_dict[layer_name]["name"])
    layer_dict = {"coef":sign_layer_list, "weight":sign_nest_dict[layer_name]["up_weight"]}

    print(layer_name)
    print(layer_dict)
    
    replace_layer(my_model, layer_name,  replace_module)
    
    G_model = copy.deepcopy(my_model)

    replace_result = validate(my_model, valid_data_loader, "cuda:0")
    print("replace_acc: " + str(replace_result[1]))

    layer_best_acc = replace_result[1] / 100.0
    layer_bast_hash = 0
    layer_best_type = "n"
    param_config = {"layer_name": layer_name,
                "ep" : group_epochs,
                "lr" : lr_c,
                "wd" : 0.01, 
                "tw" : False, 
                "twe": [], 
                "tc" : False, 
                "tce": layer_dict["coef"], 
                "do" : False,
                "lh" : 0, 
                "lt" : "n"}


    is_good = True
    train_coef = True
    need_AT = False
    improvement = True

    print_log_to_file( log_file,param_config, None, layer_best_type, layer_best_acc, None)
    while(True):
        print(" \n")
        print(param_config)
        print("good: " + str(is_good))
        print("train_coef: " + str(train_coef))
        print("need_AT: " + str(need_AT))
        print("improvement:  " + str(improvement))
        print("best_acc: " +  str(layer_best_acc))
        print(" \n")

        my_model = copy.deepcopy(G_model)
        train_result = train_group(my_model, valid_data_loader, train_data_loader, param_config, relu_list)
        train_best_acc = train_result["best_acc"]
        train_swa_acc =  train_result["swa_acc"]
        overfit = train_result["overfit"]
        train_log = train_result["train_log"]
        best_index = train_log["best_index"]

        config_backup = copy.deepcopy(param_config)
        before_acc = layer_best_acc

        if(train_best_acc > layer_best_acc or train_swa_acc > layer_best_acc):
            is_good = True
            improvement = True
            if(train_best_acc > train_swa_acc):
                layer_best_type = "b"
                layer_best_acc = train_best_acc
                G_model = copy.deepcopy(B_model)
            else:
                layer_best_type = "s"
                layer_best_acc = train_swa_acc
                G_model = copy.deepcopy(S_model)
                
            layer_bast_hash = train_result["hash"]
            param_config["lh"] = layer_bast_hash
            param_config["lt"] = layer_best_type

            if(layer_best_type == "b" and best_index < 0.5 * group_epochs):
                is_good = False
                train_coef = not train_coef
                need_AT = True

            if(not log_file == None):
                if(layer_best_type == "s"):
                    swa_log = layer_best_acc
                else:
                    swa_log = None
                print_log_to_file( log_file,config_backup, train_log, layer_best_type, before_acc, swa_log)
            
        else:
            if(is_good):
                is_good = False
                if(overfit):
                    # param_config["do"] = True
                    train_coef = not train_coef
                    need_AT = True
                else:
                    train_coef = not train_coef
                    need_AT = True
                    # param_config["lr"] = param_config["lr"] / 10
            else:
                train_coef = not train_coef
                need_AT = True

            if(len(layer_dict["weight"]) == 0):
                break

        if(need_AT):
            if(not improvement):
                break
            need_AT = False
            improvement = False
            if(train_coef):
                 param_config = {"layer_name": layer_name,
                                "ep" : group_epochs,
                                "lr" : lr_c,
                                "wd" : 0.01, 
                                "tw" : False, 
                                "twe": [], 
                                "tc" : False, 
                                "tce": layer_dict["coef"], 
                                "do" : False,
                                "lh" : layer_bast_hash, 
                                "lt" : layer_best_type}
            else:
                if(len(layer_dict["weight"]) == 0):
                    break
                param_config = {"layer_name": layer_name,
                                "ep" : group_epochs,
                                "lr" : lr_w,
                                "wd" : 0.1, 
                                "tw" : False, 
                                "twe": layer_dict["weight"], 
                                "tc" : False, 
                                "tce": [], 
                                "do" : False,
                                "lh" : layer_bast_hash, 
                                "lt" : layer_best_type}


def train_network(input_data_dirctory, model, sign_type, valid_data_loader, train_data_loader  ,start_layer_name, max_layer_counter, lr_c, lr_w):
    sign_type = sign_type
    input_data_dirctory = input_data_dirctory
    group_epochs = global_config["PA_AT"]["group_epochs"]

    folder_name = "CT_" + sign_type + "_S" + "dynamic"+"_40s/"
    
    log_directory = input_data_dirctory + "log/"
    log_file = "PR_AT_" +sign_type+".log"

    model_directory = input_data_dirctory + "model_PR_AT/"
    G_model_name = "model_PR_AT_" + sign_type +".pt"
    load_model_name = model_directory + G_model_name

    max_layer_counter = max_layer_counter
    start_layer_name = start_layer_name


    if(not os.path.exists(model_directory)):
        os.mkdir(model_directory)
    if(not os.path.exists(log_directory)):
        os.mkdir(log_directory)

    log_type = "w"
    if(start_layer_name != None):
        log_type = "a"

    with open(log_directory + log_file, log_type) as f:
        print(" ", file=f)

    print("\n")
    model = model
    sign_nest_dict = generate_sign_nest_dict(model)
    sign_param_dict = Sign_parameter_generator().param_nest_dict[sign_type]

    global G_model
    G_model = model
    if(start_layer_name != None):
        G_model = torch.load(load_model_name)
    
    resume = True
    if(start_layer_name != None):
        resume = False
    print(G_model)
    layer_counter = 0
    for key in sign_nest_dict:
        print(key)
        if(key == start_layer_name):
            resume = True
        
        if(not resume):
            continue


        file_name = key + "_coef.pt"
        scale_name = key + "_scale.pt"
        sign_dict = sign_nest_dict[key]
        sign_scale = 0
        sign_module_CT = Sign_minmax_layer(coef=torch.load(input_data_dirctory + folder_name + file_name), degree=sign_param_dict["degree"],scale=sign_scale)
        print(file_name)
        if(sign_dict["type"] == "ReLU"):
            my_layer_CT = ReLU_sign_layer(sign = sign_module_CT)
        elif(sign_dict["type"] == "MaxPool2d"):
            my_layer_CT = Maxpool_sign_layer(sign = sign_module_CT, kernel_size=sign_dict["kernel_size"], stride= sign_dict["stride"], padding=sign_dict["padding"], dilation=sign_dict["dilation"])
        else:
            raise Exception("replace layer type error")
    

        train_layer(my_model = copy.deepcopy(G_model) , layer_name = key, sign_nest_dict = sign_nest_dict,
                    valid_data_loader =  valid_data_loader, train_data_loader= train_data_loader,
                    group_epochs= group_epochs, replace_module = my_layer_CT, lr_c = lr_c, lr_w = lr_w, log_file = log_directory + log_file)
        print("Finish: " + key )
        torch.save( G_model, model_directory + G_model_name)
        layer_counter += 1
        if(layer_counter == max_layer_counter):
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str,choices=["vgg19_bn", "resnet18", "resnet32"])
    parser.add_argument("--dataset", type=str,choices=["cifar10", "imagenet", "cifar100"])
    parser.add_argument("-wd", "--working_directory", type=str, default="./working_dirctory/")
    parser.add_argument("-st","--sign_type", type=str, default="a7", choices=["a7", "2f12g1", "f1g2", "f2g2", "f2g3"])
    parser.add_argument("-sln","--start_layer_name", type=str, default="None")
    parser.add_argument("-mc","--max_counter", type=int, default=1000)
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-4)

    args = parser.parse_args()
    print(args)
    if(args.start_layer_name == "None"):
        start_layer_name = None
    else:
        start_layer_name = args.start_layer_name
    model = get_pretrained_model(model_name=args.model, dataset=args.dataset)
    lr_c = args.learning_rate
    lr_w = args.learning_rate / 10
    
    valid_data_loader = None
    train_data_loader = None 

    if(args.dataset == "cifar10" or args.dataset == "cifar100"):
        valid_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "valid", data_dir = global_config["Global"]["dataset_dirctory"])
        train_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "train", data_dir = global_config["Global"]["dataset_dirctory"])
    elif(args.dataset == "imagenet_1k"):
        valid_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "valid", data_dir = os.path.join(global_config["Global"]["dataset_dirctory"], args.dataset) )
        train_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "train", data_dir = os.path.join(global_config["Global"]["dataset_dirctory"], args.dataset) )
   
    train_network(model=model, sign_type = args.sign_type, start_layer_name = start_layer_name,
                  valid_data_loader = valid_data_loader, train_data_loader = train_data_loader,
                  max_layer_counter = args.max_counter, lr_c = lr_c, lr_w = lr_w, input_data_dirctory = args.working_directory)

