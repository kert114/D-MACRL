import time
import os
import torch
import socket
import copy
import json
from tqdm import tqdm
import numpy as np

from tensorboardX import SummaryWriter
from pprint import pprint
from datetime import datetime
from models import SimCLR
from update import LocalUpdate


from options import args_parser
from utils import exp_details, set_seed, get_dataset, save_args_json, global_repr_global_classifier, generate_erdos_renyi_graph, decentralized_average_weights, combination_matrix
from utils import initialize_psi_phi, find_phi_psi, exact_diffusion_averaging, combine_to_state_dict, average_with_A

import psutil
import csv


if __name__ == "__main__":
    start_time = time.time()

    path_project = os.path.abspath("..")    
    args = args_parser()
    exp_details(args)
    run_details = "dirich_{}_dec-{}_ED-{}_pe{}_a{}_e{}_le{}".format(args.dirichlet, args.decentralized, args.exact_diffusion, args.edge_prob, args.num_users, args.epochs, args.local_ep)
    print("Running model details ", run_details)    # define paths
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("device: ", device)
    # load dataset and user groups
    set_seed(args.seed)
    (
        train_dataset,
        test_dataset,
        user_groups,
        memory_dataset,
        test_user_groups,
    ) = get_dataset(args)
    batch_size = args.batch_size
    pprint(args)

    model_time = datetime.now().strftime("%m-%d_%H:%M") + "_{}".format(
        str(os.getpid())
    )
    model_output_dir = "save/" + model_time + run_details
    args.model_time = model_time
    save_args_json(model_output_dir, args)
    logger = SummaryWriter(model_output_dir + "/tensorboard")
    logger.add_text("Run_Details", run_details, 0)
    args.start_time = datetime.now()
    
    # build model
    start_epoch = 0
    global_model = SimCLR(args=args).to(device)
    global_weights = global_model.state_dict()

    global_model.train()
    
     # Training
    train_loss, train_accuracy, global_model_accuracy = [], [], []
    print_every = 2
    local_models = [SimCLR(args=args).to(device) for _ in range(args.num_users)]

    optimizer = torch.optim.Adam(
            global_model.parameters(), lr=args.lr, weight_decay=1e-6
        )
    
    total_epochs = int(args.epochs / args.local_ep)  # number of rounds
    
    schedule = [
    int(total_epochs * 0.3),
    int(total_epochs * 0.6),
    int(total_epochs * 0.9),
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=schedule, gamma=0.3
    )
    
    print("output model:", model_output_dir)
    print( "number of users per round: {}".format(max(int(args.frac * args.num_users), 1)))
    print("total number of rounds: {}".format(total_epochs))
    
    local_update_clients = [
        LocalUpdate(
            args=args,
            dataset=train_dataset,
            idx=idx,
            idxs=user_groups[idx],
            logger=logger,
            output_dir=model_output_dir,
        )
        for idx in range(args.num_users)
    ]
    
    for client in local_update_clients:
        client.init_model(global_model)
        
    # Create a Erdos-Renyi graph between the agents
    if args.decentralized:
        communication_graph = generate_erdos_renyi_graph(args.num_users, edge_prob=args.edge_prob)
        A = combination_matrix(communication_graph) # remove for real deal!!!!
        print("Created a Communication graph with edges = : ", communication_graph.edges())
        print("Number of edges: ", communication_graph.number_of_edges())
        print("Graph nodes: ", communication_graph.nodes())

    lr = optimizer.param_groups[0]["lr"]

    if args.exact_diffusion:
        psi, phi = initialize_psi_phi(global_model, args.num_users, total_epochs, device)
    
    psi_val = {}
    averaged_w = None
    for epoch in tqdm(range(start_epoch, total_epochs)):

        local_weights, local_losses  = [], []
        print(f"\n | Global Training Round : {epoch+1} | Model : {model_time}\n")
        if not args.decentralized:
            global_model.train()    
            
            
        for idx in range(args.num_users):
            local_model = local_update_clients[idx]
            local_model.model.train()  # Ensure the model is in training mode
            w, loss, gradients = local_model.update_ssl_weights(model=local_models[idx], global_round=epoch, lr=lr, idx=idx, M=512)
            local_models[idx] = local_model.model
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
            if args.exact_diffusion:
                psi, phi = find_phi_psi(w, gradients, idx, epoch, lr, psi, phi) 
   
        logger.add_scalar("global_train_loss", np.mean(np.array(local_losses)), epoch)
        
                
        if args.decentralized and args.average_with_A and not args.exact_diffusion:
            print("Decentralized averaging with A matrix")
            average_with_A(local_models, communication_graph, local_weights, A, args.num_users, epoch, model_output_dir)
            # print("New weights regular averaging for model are: ", A_weights)
            
        elif args.decentralized and not args.exact_diffusion:
            print("Regular Decentralized averaging ")
            for idx in range(args.num_users):
                neighbors = list(communication_graph.neighbors(idx))
                new_weights = decentralized_average_weights(local_models[idx], neighbors, local_weights, idx)
                local_models[idx] = LocalUpdate.load_weights(local_models[idx], new_weights)
                local_models[idx].save_model(model_output_dir, suffix="agent_" + str(idx), step=epoch)

        elif args.decentralized and args.exact_diffusion:
            print("Exact Diffusion")
            averaged_w = exact_diffusion_averaging(global_model, communication_graph, A, phi, epoch, args.num_users, gradients, device)
            combine_to_state_dict(local_models, averaged_w, epoch, args.num_users, model_output_dir, gradients)           
        
        else:
            print("Federated averaging")
            global_weights = LocalUpdate.average_weights(local_weights)
            for idx in range(args.num_users):
                local_models[idx] = LocalUpdate.load_weights(local_models[idx], global_weights)
            global_model.load_state_dict(global_weights)

        scheduler.step()
        lr = scheduler._last_lr[0]
        if not args.decentralized:
            global_model.save_model(model_output_dir, step=epoch)
            
    if not args.decentralized:
        global_model.save_model(model_output_dir, step=epoch)
    
    # evaluate representations
    print("evaluating representations: ", model_output_dir)
    if args.decentralized:
        print("Training a classifier on each of the local models and averaging the accuracy result")
        test_acc = 0
        for idx in range(args.num_users):
            print("Start tarining Classifier for user {}".format(idx))
            temp_test_acc = global_repr_global_classifier(args, local_models[idx], logger, args.finetuning_epoch, idx=idx)
            print("Classifier Accuracy for user {} is {}".format(idx, temp_test_acc))
            test_acc += temp_test_acc
        test_acc /= args.num_users
        print("Training a classifier on the federated global model")  
          
    else: 
        print("Training a classifier on the federated global model")
        test_acc = global_repr_global_classifier(args, global_model, logger, args.finetuning_epoch, idx=0)
       
    print(f" \n Results after {args.epochs} global rounds of training:")
    print("|----Average Classifier Test Accuracy for {} agents is: {:.2f}%".format(args.num_users ,100 * test_acc))
    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

    # PLOTTING (optional)
    pprint(args)
    suffix = "dec_{}_ED_{}_a_{}_{}_{}_{}_{}_dec_ssl_{}".format(
        args.decentralized, args.exact_diffusion, args.num_users, args.model, args.batch_size, args.epochs, args.save_name_suffix, args.ssl_method
    )
    print(suffix)




