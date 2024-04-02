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
            if(type == "s" or type == "e"):
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
        lr_scheduler = None,
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
            if(self.lr_scheduler):
                self.lr_scheduler.step(val_acc)

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
                # B_model = copy.deepcopy(self.model)
                best_epoch_i = epoch_idx
            elif(val_acc == best_val and val_loss < best_loss):
                best_val = val_acc
                best_loss = val_loss
                # B_model = copy.deepcopy(self.model)
                best_epoch_i = epoch_idx

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



def AESAP_replace(model, valid_data_loader = None):
    sign_nest_dict = generate_sign_nest_dict(model)
    print(sign_nest_dict)
    global G_model
    G_model = copy.deepcopy(model)
    for key in sign_nest_dict:
        print(key)
        if(sign_nest_dict[key]["type"] == "MaxPool2d"):
            continue

        sign_dict = sign_nest_dict[key]
        bn_name =  sign_dict["HerPN"]
        
        if(sign_dict["type"] == "ReLU" and sign_dict["HerPN"]):
            num_features = access_layer(G_model, bn_name).num_features
            BN_dimension = 2
            my_layer = HerPN2d(num_features, BN_dimension)
        else:
            print("Error: Replce Pair Can't Find")
            # assert(False, "Replce Pair Error")

        layer_name = key
        layer_dict = sign_nest_dict[key]
        replace_module = my_layer

        # print(layer_dict)

        replace_layer(model, layer_name,  replace_module)
        if(sign_nest_dict[layer_name]["HerPN"]):
            replace_layer(model, sign_nest_dict[layer_name]["HerPN"],  nn.Identity())
    
    if(valid_data_loader):
        validate(model, valid_data_loader, "cuda:0")
        
    print(model)
    return model


def AESPA_train(model, valid_data_loader, train_data_loader, config):
    layer_name = config["layer_name"]
    num_epochs = config["ep"]

    optimizer_name = "adam"
    learning_rate = config["lr"]
    weight_decay = config["wd"]
    learning_rate_decay = True
    no_bn_track = False
    my_model = model

    print("Name: " + layer_name)
    before_result  = validate(my_model, valid_data_loader, "cuda:0")

    print(layer_name +": train")

    optimizer_config = {"optimizer_type": "adam", "lr": learning_rate, "weight_decay": weight_decay}
    optimizer = get_optimizer(my_model, optimizer_config)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience = 2, eps=1e-10)
    if(learning_rate_decay):
        lr_scheduler = scheduler
    else:
        lr_scheduler = None
    trainer = Trainer(
        model=my_model,
        optimizer=optimizer,
        load_from_disk=False,
        cuda=True,
        lr_scheduler = lr_scheduler,
        no_bn_track = no_bn_track,
        train_loader = train_data_loader,
        val_loader = valid_data_loader
        
    )

    print("Layer name: " + layer_name)
    print("Parameter: ")
    print("Optimizer: " + optimizer_name)
    print("No batchnorm tracking: " +str(no_bn_track))
    print("\tLearning rate: " + str(learning_rate))
    print("\tWeight decay: " + str(weight_decay))
    print("\tLearning rate decay: " + str(learning_rate_decay))
    print("\n \n")

    print("Train epoch: ")
    train_return_pack = trainer.run_training_loop(num_epochs=num_epochs)
    train_result = validate(my_model, valid_data_loader, "cuda:0")
    global E_model
    E_model = copy.deepcopy(my_model)
    print("\n \n")
    print("Validation result:")
    print("\tbefore: " + str(before_result))
    print("\ttrain: "+ str(train_result))
    print("\n \n \n \n")





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str,choices=["vgg19_bn", "resnet18", "resnet32", "resnet20"])
    parser.add_argument("--dataset", type=str,choices=["cifar10", "imagenet_1k", "cifar100"])
    parser.add_argument("-wd", "--working_directory", type=str, default="./working_dirctory/")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-6)

    args = parser.parse_args()
    print(args)
    valid_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "valid", data_dir = global_config["Global"]["dataset_dirctory"])
    train_data_loader = get_data_loader(dataset = args.dataset, dataset_type = "train", data_dir = global_config["Global"]["dataset_dirctory"])
    model = get_pretrained_model(model_name=args.model, dataset=args.dataset)
    print(model)
    validate(model, valid_data_loader, "cuda:0")
    AESAP_replace(model)
    lr_c = args.learning_rate
    param_config = {"layer_name": "Whole Model",
            "ep" : 5,
            "lr" : lr_c,
            "wd" : 0.01}
    AESPA_train(model, valid_data_loader, train_data_loader, param_config)

