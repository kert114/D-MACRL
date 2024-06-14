import copy
import torch
import numpy as np
import torch.nn as nn
import time
import os
import csv
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(
        self, dataset, idxs, idx=0, noniid=False, noniid_prob=1.0, xshift_type="rot"
    ):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.idx = idx
        self.noniid = noniid
        self.classes = self.dataset.classes
        self.targets = np.array(self.dataset.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdate(object):
    def __init__(
        self,
        args,
        dataset,
        idx,
        idxs,
        logger=None,
        test_dataset=None,
        memory_dataset=None,
        output_dir="",
    ):
        self.args = args
        self.logger = logger
        self.id = idx  # user id
        self.idxs = idxs  # dataset id
        self.reg_scale = args.reg_scale
        self.output_dir = output_dir
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self.criterion = nn.NLLLoss().to(self.device)

        self.logger = logger
        
        if dataset is not None:
            self.trainloader, self.validloader, self.testloader = self.train_val_test(
                dataset, list(idxs), test_dataset, memory_dataset
            )
    
    def init_model(self, model):
        """Initialize local models"""
        train_lr = self.args.lr
        self.model = copy.deepcopy(model)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_lr, weight_decay=1e-6
        )

        total_epochs = self.args.local_ep * self.args.epochs
        self.schedule = [
            int(total_epochs * 0.3),
            int(total_epochs * 0.6),
            int(total_epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.schedule, gamma=0.3
        )
        self.scheduler = scheduler
        self.optimizer = optimizer

    
    def train_val_test(self, dataset, idxs, test_dataset=None, memory_dataset=None):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes. split indexes for train, validation, and test (80, 10, 10)
        """
        train_to = 0.01
        val_from = 0.89
        test_from = 0.9
        print("Training uses {} andd validation {} of the dataset".format(train_to, test_from - val_from))
        idxs_train = idxs[: int(train_to * len(idxs))]
        self.idxs_train = idxs_train
        idxs_val = idxs[int(val_from * len(idxs)) : int(0.9 * len(idxs))]
        idxs_test = idxs[int(test_from * len(idxs)) :]
        print("idx train: ", len(idxs_train))

        train_dataset = DatasetSplit(
            dataset,
            idxs_train,
            idx=self.id,
        )
        
        trainloader = DataLoader(
                train_dataset,
                batch_size=self.args.local_bs,
                shuffle=True,
                num_workers=16, # was 16
                pin_memory=True,
                drop_last=True if len(train_dataset) > self.args.local_bs else False,
            )

        validloader = DataLoader(
            DatasetSplit(
                dataset,
                idxs_val,
                idx=self.id,
            ),
            batch_size=self.args.local_bs,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        testloader = DataLoader(
            DatasetSplit(
                dataset,
                idxs_test,
                idx=self.id,
            ),
            batch_size=64,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        if test_dataset is not None:
            # such that the memory loader is the original dataset without pair augmentation
            memoryloader = DataLoader(
                DatasetSplit(
                    memory_dataset,
                    idxs_train,
                    idx=self.id,
                ),
                batch_size=64,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )

        else:
            memoryloader = DataLoader(
                DatasetSplit(
                    dataset,
                    idxs_train,
                    idx=self.id,
                ),
                batch_size=self.args.local_bs,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )

        self.memory_loader = memoryloader
        self.test_loader = testloader

        return trainloader, validloader, testloader
        
    def update_ssl_weights(
        self,
        model,
        global_round,
        additionl_feature_banks=None,
        lr=None,
        epoch_num=None,
        vis_feature=False,
        idx=None,
        M=None
    ):
        """Train the local model with self-superivsed learning"""
        
        print("Updating local model for agent: ", idx)
        
        epoch_loss = [0]
        global_model_copy = copy.deepcopy(model)
        global_model_copy.eval()

        # Set optimizer for the local updates
        train_epoch = epoch_num if epoch_num is not None else self.args.local_ep
        
        train_lr = lr if lr is not None else self.args.lr
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_lr, weight_decay=1e-6
        )
        
        schedule = [
            int(self.args.local_ep * self.args.epochs * 0.3),
            int(self.args.local_ep * self.args.epochs * 0.6),
            int(self.args.local_ep * self.args.epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule, gamma=0.3
        )
        
        global_step = 0
        max_steps = len(self.trainloader) * self.args.local_ep
        
        train_epoch_ = int(np.ceil(train_epoch))
        max_iter = int(train_epoch * len(self.trainloader))
        epoch_start_time = time.time()
        
        for iter in range(train_epoch_):
            model.train()
            local_curr_ep = self.args.local_ep * global_round + iter
        
            batch_loss = []
            batch_size = self.args.local_bs
            temperature = self.args.temperature
            start_time = time.time()
        
            for batch_idx, data in enumerate(self.trainloader):
                data_time = time.time() - start_time
                start_time = time.time()

                (pos_1, pos_2, labels) = data
                loss, feat = model(
                    pos_1.to(self.device),
                    pos_2.to(self.device),
                    return_feat=True,
                )
            
                loss = loss.mean()
                optimizer.zero_grad()
                if not loss.isnan().any():
                    loss.backward()
                    optimizer.step()

                
                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.data.clone()  # Store a copy of the gradient
                
                model_time = time.time() - start_time
                start_time = time.time()
                
                # self.logger.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], global_round * self.args.local_ep + iter)
                    
                
                if batch_idx % 10 == 0:
                    print(
                        "Update SSL || User : {} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f} \
                        LR: {:.4f}  Feat: {:.3f} Epoch Time: {:.3f} Model Time: {:.3f} Data Time: {:.3f} Model: {}".format(
                            self.id,
                            global_round,
                            local_curr_ep,
                            batch_idx * len(labels),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                            optimizer.param_groups[0]["lr"],
                            feat.mean().item(),
                            time.time() - epoch_start_time,
                            model_time,
                            data_time,
                            self.args.model_time,
                        )
                    )
                    
                # self.logger.add_scalar("loss", loss.item())
                self.logger.add_scalar(f"Train_Loss/Agent_{idx}", loss.item(), global_step=global_round * self.args.local_ep + iter)
                batch_loss.append(loss.item())
                data_start_time = time.time()
                scheduler.step()
        
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            
        self.model = model
        self.optimizer = optimizer
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), gradients

    def average_weights(w, avg_weights=None):
        """
        Returns the average of the weights.
        """
        w_avg = copy.deepcopy(w[0])
        for key in w[0].keys():
            for i in range(1, len(w)):
                w_avg[key] = w_avg[key] + w[i][key]

            w_avg[key] = torch.div(w_avg[key], len(w))
        print("len of w: ", len(w))
        return w_avg
    
    def load_weights(model, w):
        """
        Returns the average of the weights.
        """
        model.load_state_dict({k: v for k, v in w.items()}, strict=False)
        return model
    
    def test_inference(args, model, test_dataset):
        """Returns the test accuracy and loss."""

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        criterion = nn.NLLLoss().to(device)
        testloader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False
        )

        test_bar = tqdm(testloader, desc="Linear Probing", mininterval=1, ascii=True)

        for (images, labels) in test_bar:
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            test_bar.set_postfix({"Accuracy": correct / total * 100})

        accuracy = correct / total
        return accuracy, loss