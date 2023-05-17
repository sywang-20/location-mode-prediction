import sys, os
import pandas as pd
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score

import time

from transformers import get_linear_schedule_with_warmup

from utils.earlystopping import EarlyStopping
from utils.dataloader import load_pk_file


def get_performance_dict(return_dict):
    perf = {
        "correct@1": return_dict["correct@1"],
        "correct@3": return_dict["correct@3"],
        "correct@5": return_dict["correct@5"],
        "correct@10": return_dict["correct@10"],
        "rr": return_dict["rr"],
        "f1": return_dict["f1"],
        "total": return_dict["total"],
    }

    perf["acc@1"] = perf["correct@1"] / perf["total"] * 100
    perf["acc@5"] = perf["correct@5"] / perf["total"] * 100
    perf["acc@10"] = perf["correct@10"] / perf["total"] * 100
    perf["mrr"] = perf["rr"] / perf["total"] * 100

    return perf


def send_to_device(inputs, device, config):
    x, y, x_dict, y_mode = inputs
    if config.networkName == "deepmove":
        x = (x[0].to(device), x[1].to(device))
        
        for key in x_dict[0]:
            x_dict[0][key] = x_dict[0][key].to(device)
        for key in x_dict[1]:
            x_dict[1][key] = x_dict[1][key].to(device)
    else:
        x = x.to(device) # move to GPU
        for key in x_dict:
            x_dict[key] = x_dict[key].to(device)
    y = y.to(device)
    y_mode = y_mode.to(device)

    return x, y, x_dict, y_mode


def calculate_correct_total_prediction(logits, true_y):

    # top_ = torch.eq(torch.argmax(logits, dim=-1), true_y).sum().cpu().numpy()

    result_ls = []
    for k in [1, 3, 5, 10]:
        if logits.shape[-1] < k:
            prediction = torch.argmax(logits, dim=-1)
        else:
            prediction = torch.topk(logits, k=k, dim=-1).indices
        # f1 score
        if k == 1:
            f1 = f1_score(true_y.cpu(), prediction.cpu(), average="weighted")

        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum().cpu().numpy()
        # top_k = np.sum([curr_y in pred for pred, curr_y in zip(prediction, true_y)])
        result_ls.append(top_k)

    # f1 score
    result_ls.append(f1)
    # mrr
    result_ls.append(get_mrr(logits, true_y))
    # total
    result_ls.append(true_y.shape[0])

    return np.array(result_ls, dtype=np.float32)


def get_mrr(prediction, targets):
    """
    Calculates the MRR score for the given predictions and targets.

    Args:
        prediction (Bxk): torch.LongTensor. the softmax output of the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        the sum rr score
    """
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float()
    rranks = torch.reciprocal(ranks)

    return torch.sum(rranks).cpu().numpy()


def get_optimizer(config, model):
    # define the optimizer & learning rate
    if config.optimizer == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=True,
        )
    elif config.optimizer == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

    return optim


def trainNet(config, model, train_loader, val_loader, device, log_dir):

    performance = {}
    # define the optimizer
    optim = get_optimizer(config, model)

    # define learning rate schedule
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,
        num_training_steps=len(train_loader) * config.num_training_epochs,
    )
    scheduler_ES = StepLR(optim, step_size=config.lr_step_size, gamma=config.lr_gamma)
    if config.verbose:
        print("Current learning rate: ", scheduler.get_last_lr()[0])

    # Time for printing
    training_start_time = time.time()
    globaliter = 0
    scheduler_count = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config["patience"], verbose=config.verbose)

    # Loop for n_epochs
    for epoch in range(config.max_epoch):
        # train for one epoch
        globaliter = train(
            config,
            model,
            train_loader,
            optim,
            device,
            epoch,
            scheduler,
            scheduler_count,
            globaliter,
        )

        # At the end of the epoch, do a pass on the validation set
        return_dict = validate(config, model, val_loader, device)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(return_dict, model)

        if early_stopping.early_stop:
            if config.verbose:
                print("=" * 50)
                print("Early stopping")
            if scheduler_count == 2:
                performance = get_performance_dict(early_stopping.best_return_dict)
                print(
                    "Training finished.\t Time: {:.2f}min.\t acc@1: {:.2f}%".format(
                        (time.time() - training_start_time) / 60,
                        performance["acc@1"],
                    )
                )

                break

            scheduler_count += 1
            model.load_state_dict(torch.load(log_dir + "/checkpoint.pt"))
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler_ES.step()

        if config.verbose:
            # print("Current learning rate: {:.5f}".format(scheduler.get_last_lr()[0]))
            # print("Current learning rate: {:.5f}".format(scheduler_ES.get_last_lr()[0]))
            print("Current learning rate: {:.5f}".format(optim.param_groups[0]["lr"]))
            print("=" * 50)

        if config.debug == True:
            break

    return model, performance


def train(
    config,
    model,
    train_loader,
    optim,
    device,
    epoch,
    scheduler,
    scheduler_count,
    globaliter,
):

    model.train() # set model to training mode

    running_loss = 0.0
    # 1, 3, 5, 10, rr, total
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    n_batches = len(train_loader)

    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)  # cross entropy loss
    # define start time
    start_time = time.time()
    optim.zero_grad()  # clear gradients before computing the gradient for a new epoch
    for i, inputs in enumerate(train_loader):   # i: batch index inputs: batch data
        globaliter += 1

        x, y, x_dict, y_mode = send_to_device(inputs, device, config)  # send data to device

        if config.if_loss_mode:
            # whether to use the mode loss
            if config.if_embed_next_mode:
                # whether to embed the next mode
                logits_loc, logits_mode = model(x, x_dict, device, next_mode=y_mode) # n+1时刻的结果
            else:
                logits_loc, logits_mode = model(x, x_dict, device)
            # calculate the loss
            loss_size_loc = CEL(logits_loc, y.reshape(-1))
            loss_size_mode = CEL(logits_mode, y_mode.reshape(-1))
            # add the loss (location + mode)
            loss_size = loss_size_loc + loss_size_mode
        else:
            logits_loc = model(x, x_dict, device)
            loss_size = CEL(logits_loc, y.reshape(-1))
        # backward, update parameters during training
        optim.zero_grad() # clear gradients before computing the gradient for a new batch
        loss_size.backward() # compute gradients using backpropagation--> error

        # clip the gradient, avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # update parameters
        optim.step()
        if scheduler_count == 0:
            scheduler.step() # update learning rate

        # Print statistics
        running_loss += loss_size.item()

        result_arr += calculate_correct_total_prediction(logits_loc, y)

        if (config.verbose) and ((i + 1) % config["print_step"] == 0):
            print(
                "Epoch {}, {:.1f}%\t loss: {:.3f} acc@1: {:.2f} f1: {:.2f} mrr: {:.2f}, took: {:.2f}s \r".format(
                    epoch + 1,
                    100 * (i + 1) / n_batches,
                    running_loss / config["print_step"],
                    100 * result_arr[0] / result_arr[-1],
                    100 * result_arr[4] / config["print_step"],
                    100 * result_arr[5] / result_arr[-1],
                    time.time() - start_time,
                ),
                end="",
                flush=True,
            )

            # Reset running loss and time
            running_loss = 0.0
            result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            start_time = time.time()

        if (config["debug"] == True) and (i > 20):
            break
    if config.verbose:
        print()
    return globaliter


def validate(config, model, data_loader, device):

    total_val_loss = 0
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for inputs in data_loader:

            x, y, x_dict, y_mode = send_to_device(inputs, device, config)

            if config.if_loss_mode:
                if config.if_embed_next_mode:
                    logits_loc, logits_mode = model(x, x_dict, device, next_mode=y_mode)
                else:
                    logits_loc, logits_mode = model(x, x_dict, device)

                loss_size_loc = CEL(logits_loc, y.reshape(-1))
                loss_size_mode = CEL(logits_mode, y_mode.reshape(-1))

                loss_size = loss_size_loc + loss_size_mode
            else:
                logits_loc = model(x, x_dict, device)
                loss_size = CEL(logits_loc, y.reshape(-1))

            total_val_loss += loss_size.item()

            result_arr += calculate_correct_total_prediction(logits_loc, y.view(-1))

    val_loss = total_val_loss / len(data_loader)
    result_arr[4] = result_arr[4] / len(data_loader)

    if config.verbose:
        print(
            "Validation loss = {:.2f} acc@1 = {:.2f} f1 = {:.2f} mrr = {:.2f}".format(
                val_loss,
                100 * result_arr[0] / result_arr[-1],
                100 * result_arr[4],
                100 * result_arr[5] / result_arr[-1],
            ),
        )

    return {
        "val_loss": val_loss,
        "correct@1": result_arr[0],
        "correct@3": result_arr[1],
        "correct@5": result_arr[2],
        "correct@10": result_arr[3],
        "f1": result_arr[4],
        "rr": result_arr[5],
        "total": result_arr[6],
    }


def test(config, model, data_loader, device):
    # overall accuracy
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    # per user accuracy
    result_dict = {}
    batch_dict = {}
    for i in range(1, config.total_user_num):
        result_dict[i] = {}
        batch_dict[i] = {}
        for j in range(1, 8):
            result_dict[i][j] = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            batch_dict[i][j] = 0

    # change to validation mode
    model.eval()
    with torch.no_grad():

        for inputs in data_loader:
            x, y, x_dict, y_mode = send_to_device(inputs, device, config)

            if config.if_loss_mode:
                if config.if_embed_next_mode:
                    logits_loc, _ = model(x, x_dict, device, next_mode=y_mode)
                else:
                    logits_loc, _ = model(x, x_dict, device)
            else:
                logits_loc = model(x, x_dict, device)

            # we get the per user per mode accuracy
            user_arr = x_dict["user"].cpu().detach().numpy()
            mode_arr = y_mode.cpu().detach().numpy()
            for user in np.unique(user_arr):
                # index belong to the current user 
                user_index = np.nonzero(user_arr == user)[0]
                for mode in np.unique(mode_arr):
                    mode_index = np.nonzero(mode_arr == mode)[0]
                    index = set(user_index).intersection(set(mode_index))
                    if not len(index):
                        continue
                    index = np.array(list(index))
                    result_dict[user][mode] += calculate_correct_total_prediction(logits_loc[index, :], y[index])
                    batch_dict[user][mode] += 1

            result_arr += calculate_correct_total_prediction(logits_loc, y.view(-1))

    # f1 score
    for i in range(1, config.total_user_num):
        for j in range(1, 8):
            if batch_dict[i][j] != 0:
                result_dict[i][j][4] = result_dict[i][j][4]/batch_dict[i][j]

    result_arr[4] = result_arr[4] / len(data_loader)

    if config.verbose:
        print(
            "acc@1 = {:.2f} f1 = {:.2f} mrr = {:.2f}".format(
                100 * result_arr[0] / result_arr[-1],
                100 * result_arr[4],
                100 * result_arr[5] / result_arr[-1],
            ),
        )

    return (
        {
            "correct@1": result_arr[0],
            "correct@3": result_arr[1],
            "correct@5": result_arr[2],
            "correct@10": result_arr[3],
            "f1": result_arr[4],
            "rr": result_arr[5],
            "total": result_arr[6],
        },
        result_dict
    )
