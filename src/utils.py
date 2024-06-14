import os
import torch
import numpy as np
import random
import time
import json
import os
import csv
import torch.nn as nn
import networkx as nx
import copy
import psutil
from tqdm import tqdm


from datetime import datetime
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from update import LocalUpdate
from sampling import cifar_iid, cifar_noniid_dirichlet, cifar_split_dataset


modified_cifar_data = None
modified_cifar_test_data = None


def generate_erdos_renyi_graph(num_users, edge_prob):
    if edge_prob == 0:
        return nx.erdos_renyi_graph(n=num_users, p=edge_prob)
    
    while True:
        # Generate an Erdős-Rényi graph
        G = nx.erdos_renyi_graph(n=num_users, p=edge_prob)
        G_matrix = nx.to_numpy_array(G)
        print("attempted matrix ", G_matrix)
        # Check if the generated graph is connected
        if nx.is_connected(G):
            print('successful matrix ', G_matrix)
            return G
 
def combination_matrix(G):
    """Constructs the Metropolis rule combination matrix from a NetworkX graph."""
    # Convert graph to adjacency matrix as a numpy array
    C = nx.to_numpy_array(G)
    np.fill_diagonal(C, 1)

    K = C.shape[0]  # Number of nodes
    n = C @ np.ones((K,))
    
    # Initialize the combination matrix A with zeros
    A = np.zeros((K, K))
    
    # Compute off-diagonal elements according to Metropolis rule
    for k in range(K):
        for l in range(k + 1, K):  # Only upper triangle needed due to symmetry
            if C[k, l] == 1:
                A[k, l] = np.true_divide(1, np.max([n[k], n[l]]))
                print(np.max([n[k], n[l]]))
                A[l, k] = A[k, l]

    # Compute diagonal elements to make rows sum to 1
    degrees = A @ np.ones((K,))
    for k in range(K):
        A[k, k] = 1 - degrees[k]
        
    eigs = np.linalg.eigvalsh(A)
    lambda_2 = eigs[-2]
    print("matrix A ", A)
    print('Second Eigenvalue of A: ', lambda_2)
    return A

def initialize_psi_phi(model, num_users, total_epochs, device):
    psi = {}
    phi = {}
    state_dict = model.state_dict()
    total_epochs = 2# Get the model's initial state dict

    # Initialize psi and phi with the same structure as the model's state_dict
    for key, param in state_dict.items():
        # Each parameter in psi and phi should match the shape of the model's parameters
        # We add extra dimensions for num_users and total_epochs to handle all epochs and users
        psi[key] = torch.zeros((num_users, total_epochs, *param.shape), device=device)
        phi[key] = torch.zeros_like(psi[key])
    
    return psi, phi

def initialize_temp_phi(model, num_users, device):
    phi = {}
    state_dict = model.state_dict() 

    for key, param in state_dict.items():
        phi[key] = torch.zeros((num_users, *param.shape), device=device)

    
    return phi


def average_weights(w, avg_weights=None):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key]

        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_with_A(local_models, communication_graph, all_weights, A, num_users, epoch, model_output_dir):
    print("Decentralized averaging with A")

    w_avg = {idx: {key: torch.zeros_like(value, dtype=torch.float) for key, value in all_weights[0].items()} for idx in range(num_users)}

    for idx in range(num_users):
        agent_weight = copy.deepcopy(local_models[idx].state_dict())
        neighbors = list(communication_graph.neighbors(idx))

        for key in w_avg[idx].keys():
            # Accumulate weighted weights from self and neighbors
            for neighbor in [idx] + neighbors:
                w_avg[idx][key] += A[idx, neighbor] * all_weights[neighbor][key]
            
        local_models[idx] = LocalUpdate.load_weights(local_models[idx], w_avg[idx])
        local_models[idx].save_model(model_output_dir, suffix="agent_" + str(idx), step=epoch)


def decentralized_average_weights(local_model, neighbors, all_weights, idx):
    agent_weights = copy.deepcopy(local_model.state_dict())

    for neighbor in neighbors:
        neighbor_weights = all_weights[neighbor]
        for key in agent_weights:
            agent_weights[key] += neighbor_weights[key]
    for key in agent_weights:
        agent_weights[key] = agent_weights[key].float() 
        agent_weights[key] /= (len(neighbors) + 1)
        
    return agent_weights

def updated_find_phi_psi(w, gradients, idx,  epoch, psi, phi, prev_w):    
    for key in w:
        if key in gradients:
            if epoch != 0:
                psi[key][idx, 0, :] = psi[key][idx, 1, :]
                
            psi[key][idx, 1, :] = w[key] 
            if epoch == 0:
                phi[key][idx, 1, :] = psi[key][idx, 1, :]
            else:
                phi[key][idx, 1, :] = psi[key][idx, 1, :] + prev_w[key][idx] - psi[key][idx, 0, :]
                
    return psi, phi

def find_phi_psi(w, gradients, idx, epoch, lr, psi, phi):    
    print("Find_phi_psi for agent: ", idx)
    mu = lr    
    print("mu: ", mu)
    for key in w:
        if key in gradients:
            psi[key][idx, epoch, :] = w[key] - mu * gradients[key]
            
            if epoch == 0:
                phi[key][idx, epoch, :] = psi[key][idx, epoch, :]
            else:
                phi[key][idx, epoch, :] = psi[key][idx, epoch, :] + w[key] - psi[key][idx, epoch - 1, :]
        
    return psi, phi

            
def exact_diffusion_averaging(local_model, communication_graph, A, phi, epoch, num_users, gradients, device):
    
    print("exact_diffusion_averaging")
    temp_w = initialize_temp_phi(local_model, num_users, device)
    for idx in range(num_users):
        neighbors = list(communication_graph.neighbors(idx))
        for key in phi.keys():
            if key in gradients:
                for neighbor in [idx] + neighbors:
                    temp_w[key][idx] += A[idx, neighbor] * phi[key][neighbor, 1]
    return temp_w


def combine_to_state_dict(local_models, averaged_phi, epoch, num_users, model_output_dir, gradients):
    print("combine_to_state_dict between {} agents".format(num_users))
    for idx in range(num_users):
        local_model = local_models[idx]
        state_dict = local_model.state_dict()

        for key in state_dict.keys():
            if key in averaged_phi and key in gradients:
                state_dict[key] = averaged_phi[key][idx]

        local_model = LocalUpdate.load_weights(local_model, state_dict)
        local_model.save_model(model_output_dir, suffix=f"agent_{idx}", step=epoch)
 

def get_dataset(args, **kwargs):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    global modified_cifar_data, modified_cifar_test_data
    start = time.time()
    print("sampling for dataset: {}".format(args.dataset))


    data_dir = "data/cifar/"
    dataset_name = "CIFAR10"
    train_dataset_func = CIFAR10Pair
    img_size = 32
    train_transform_ = get_transform(img_size)
    test_transform_ = test_transform

    train_dataset = train_dataset_func(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform_,
    )

    print("train dataset sample num:", train_dataset.data.shape)
    test_dataset = getattr(datasets, dataset_name)(
        data_dir, train=False, download=True, transform=test_transform_
    )
    memory_dataset = getattr(datasets, dataset_name)(
        data_dir, train=True, download=True, transform=test_transform_
    )
    
    print("get dataset time: {:.3f}".format(time.time() - start))
    start = time.time()
        
         # sample training data among users
         
    if args.iid:
        print("Use i.i.d. sampling")
        user_groups = cifar_iid(train_dataset, args.num_users)
        test_user_groups = cifar_iid(test_dataset, args.num_users)
    
    if args.split_dataset != None:
        user_groups = cifar_split_dataset(train_dataset, args.num_users, args.split_dataset)
        test_user_groups = cifar_split_dataset(test_dataset, args.num_users, args.split_dataset)

        
    else:
        if args.dirichlet:
            print("Y dirichlet sampling")
            user_groups = cifar_noniid_dirichlet(
                train_dataset, args.num_users, args.dir_beta, vis=True
            )
            test_user_groups = cifar_noniid_dirichlet(
                test_dataset, args.num_users, args.dir_beta
            )
            
        else:
            print("Use i.i.d. sampling")
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)
            
        
    print("sample dataset time: {:.3f}".format(time.time() - start))
    print(
        "user data samples:", [len(user_groups[idx]) for idx in range(len(user_groups))]
    )
    return train_dataset, test_dataset, user_groups, memory_dataset, test_user_groups

def exp_details(args):
    print("\nExperimental details:")
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.epochs}\n")
    print(f"    Fraction of users  : {args.frac}")
    print(f"    Local Batch size   : {args.local_bs}")
    print(f"    Local Epochs       : {args.local_ep}\n")
    return

def get_dist_env():
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE"))
    else:
        world_size = int(os.getenv("SLURM_NTASKS"))

    if "OMPI_COMM_WORLD_RANK" in os.environ:
        global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK"))
    else:
        global_rank = int(os.getenv("SLURM_PROCID"))
    return global_rank, world_size

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

def save_args_json(path, args):
    mkdir_if_missing(path)
    arg_json = os.path.join(path, "args.json")
    with open(arg_json, "w") as f:
        args = vars(args)
        json.dump(args, f, indent=4, sort_keys=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
def write_log_and_plot(
        model_time, model_output_dir, args, suffix, test_acc, intermediate=False
    ):
        # mkdir_if_missing('save')
        if not os.path.exists("save/" + args.log_directory):
            os.makedirs("save/" + args.log_directory)

        log_file_name = (
            args.log_file_name + "_intermediate" if intermediate else args.log_file_name
        )
        elapsed_time = (
            (datetime.now() - args.start_time).seconds if hasattr(args, "start_time") else 0
        )
        with open(
            "save/{}/best_linear_statistics_{}.csv".format(
                args.log_directory, log_file_name
            ),
            "a+",
        ) as outfile:
            writer = csv.writer(outfile)

            res = [
                suffix,
                "",
                "",
                args.dataset,
                "acc: {}".format(test_acc),
                "num of user: {}".format(args.num_users),
                "frac: {}".format(args.frac),
                "epoch: {}".format(args.epochs),
                "local_ep: {}".format(args.local_ep),
                "local_bs: {}".format(args.local_bs),
                "lr: {}".format(args.lr),
                "backbone: {}".format(args.backbone),
                "dirichlet {}: {}".format(args.dirichlet, args.dir_beta),
                "imagenet_based_cluster: {}".format(args.imagenet_based_cluster),
                "partition_skew: {}".format(args.y_partition_skew),
                "partition_skew_ratio: {}".format(args.y_partition_ratio),
                "iid: {}".format(args.iid),
                "reg scale: {}".format(args.reg_scale),
                "cont opt: {}".format(args.model_continue_training),
                model_time,
                "elapsed_time: {}".format(elapsed_time),
            ]
            writer.writerow(res)

            name = "_".join(res).replace(": ", "_")
            print("writing best results for {}: {} !".format(name, test_acc))
            
def _get_classifier_dataset(args):
    if args.dataset.endswith("ssl"):
        args.dataset = args.dataset[:-3]  # remove the ssl
    train_dataset, test_dataset, _, _, _ = get_dataset(args)
    return train_dataset, test_dataset


def global_repr_global_classifier(args, model, logger, test_epoch=60, idx=None): # test_epoch=60
    from models import ResNetCifarClassifier
    from update import LocalUpdate   
    print("Training classifier")
    # global representation, global classifier
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    train_dataset, test_dataset = _get_classifier_dataset(args)

    print("begin training classifier...")
    model_classifier = None
    model_classifier = ResNetCifarClassifier(args=args)
    # if hasattr(global_model, "module"):
    #     global_model = global_model.module
    model_classifier.load_state_dict(
        model.state_dict(), strict=False
    )  
    model_classifier = model_classifier.to(device)
    for param in model_classifier.f.parameters():
        param.requires_grad = False

    # train only the last layer
    optimizer = torch.optim.Adam(
        model_classifier.fc.parameters(), lr=1e-3, weight_decay=1e-6
    )

    
    
    trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16, pin_memory=True)
    # testloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

    criterion = torch.nn.NLLLoss().to(device)
    
    best_acc = 0
    # train global model on global dataset
    for epoch_idx in tqdm(range(test_epoch)):
        batch_loss = []
        
        for batch_idx, (images, _, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(
                    "Downstream Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch_idx + 1,
                        batch_idx * len(images),
                        len(trainloader.dataset),
                        100.0 * batch_idx / len(trainloader),
                        loss.item(),
                    )
                )
            batch_loss.append(loss.item())
            logger.add_scalar("Classifier_train_loss", loss.item(), epoch_idx * len(trainloader) + batch_idx)
            logger.add_scalar(f"ClassifierTrainingLoss/agent_{idx}", loss.item(), epoch_idx * len(trainloader) + batch_idx)
            # print(f"Model {idx} Epoch {epoch_idx+1} Batch {batch_idx}: Loss {loss.item()}")
            # logger.add_scalar("Classifier_train_loss", loss.item(), epoch_idx * len(trainloader) + batch_idx)

        loss_avg = sum(batch_loss) / len(batch_loss)
        test_acc, test_loss = LocalUpdate.test_inference(args, model_classifier, test_dataset)
        logger.add_scalar(f"ClassifierTestLoss/agent_{idx}", test_loss, epoch_idx)
        logger.add_scalar(f"ClassifierTestAccuracy/agent_{idx}", test_acc, epoch_idx)
        logger.add_scalar("Classifier_test_loss", test_loss, epoch_idx)
        logger.add_scalar("Classifier_test_accuracy", test_acc, epoch_idx)
        if test_acc > best_acc:
            best_acc = test_acc
        print("\n Downstream Train loss: {} Acc: {}".format(loss_avg, best_acc))
    return best_acc


def write_log_and_plot(
    model_time, model_output_dir, args, suffix, test_acc, intermediate=False
):
    # mkdir_if_missing('save')
    if not os.path.exists("save/" + args.log_directory):
        os.makedirs("save/" + args.log_directory)

    log_file_name = (
        args.log_file_name + "_intermediate" if intermediate else args.log_file_name
    )
    elapsed_time = (
        (datetime.now() - args.start_time).seconds if hasattr(args, "start_time") else 0
    )
    with open(
        "save/{}/best_linear_statistics_{}.csv".format(
            args.log_directory, log_file_name
        ),
        "a+",
    ) as outfile:
        writer = csv.writer(outfile)

        res = [
            suffix,
            "",
            "",
            args.dataset,
            "acc: {}".format(test_acc),
            "num of user: {}".format(args.num_users),
            "frac: {}".format(args.frac),
            "epoch: {}".format(args.epochs),
            "local_ep: {}".format(args.local_ep),
            "local_bs: {}".format(args.local_bs),
            "lr: {}".format(args.lr),
            "backbone: {}".format(args.backbone),
            "dirichlet {}: {}".format(args.dirichlet, args.dir_beta),
            "imagenet_based_cluster: {}".format(args.imagenet_based_cluster),
            "partition_skew: {}".format(args.y_partition_skew),
            "partition_skew_ratio: {}".format(args.y_partition_ratio),
            "iid: {}".format(args.iid),
            "reg scale: {}".format(args.reg_scale),
            "cont opt: {}".format(args.model_continue_training),
            model_time,
            "elapsed_time: {}".format(elapsed_time),
        ]
        writer.writerow(res)

        name = "_".join(res).replace(": ", "_")
        print("writing best results for {}: {} !".format(name, test_acc))
        
import torch.nn as nn
from torchvision.models import resnet50, resnet18

def get_backbone(pretrained_model_name="resnet50", weights=None, full_size=False):
    # Dictionary mapping model names to model functions
    model_dict = {
        'resnet50': resnet50,
        'resnet18': resnet18
    }

    # Retrieve the model function based on the model name
    model_func = model_dict.get(pretrained_model_name)
    if model_func is None:
        raise ValueError("Unsupported model name: {}".format(pretrained_model_name))
    print("Backbone model function: ", model_func)
    # Instantiate the model using the 'weights' parameter
    model = model_func(weights=weights)

    f = []
    for name, module in model.named_children():
        if name == "conv1" and not full_size:
            module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        if full_size:
            if name != "fc":
                f.append(module)
        else:
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                f.append(module)

    # Create a sequential module from the selected layers
    f = nn.Sequential(*f)

    # Set the feature dimension based on the model type
    feat_dim = 2048 if "resnet50" in pretrained_model_name else 512

    return f, feat_dim

    
get_transform = lambda s: transforms.Compose(
    [
        transforms.RandomResizedCrop(s),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p = 0.2), # added gaussian blur
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)

get_transform_mae = lambda s: transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)
class CIFAR10Pair(datasets.CIFAR10):
    """CIFAR10 Dataset."""

    def __init__(
        self,
        class_id=None,
        tgt_class=None,
        sample_num=10000,
        imb_factor=1,
        imb_type="",
        with_index=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_num = sample_num

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return pos_1, pos_2, target
    
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Current memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")  # RSS is the Resident Set Size

