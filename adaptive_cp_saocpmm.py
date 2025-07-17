import argparse
from collections import defaultdict
import math
import os
from re import sub
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, LBFGS, SGD
import tqdm
import statistics

from online_conformal.saocp_upd import SAOCP_UPD
from online_conformal.saocp import SAOCP
from online_conformal.samocp import SAMOCP
from online_conformal.mocp import MOCP
from online_conformal.faci import FACI, FACI_S
from online_conformal.ogd import ScaleFreeOGD
from online_conformal.split_conformal import SplitConformal
from online_conformal.utils import pinball_loss,quantile,pinball_loss_new
from cv_utils import create_model, data_loader
from cv_utils import ImageNet, TinyImageNet, CIFAR10, CIFAR100, ImageNetC, TinyImageNetC, CIFAR10C, CIFAR100C

import csv


corruptions = [
    None,
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Runs conformal prediction experiments on computer vision datasets. If you want to do multi-GPU "
        f"training, call this file with `torchrun --nproc_per_node <ngpu> {os.path.basename(__file__)} ...`."
        f"But if training is finished, we recommend not doing this."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ImageNet", "TinyImageNet", "CIFAR10", "CIFAR100"],
        help="Dataset to run on.",
    )

    parser.add_argument(
        "--distribution_shift",
        required=True,
        choices=["gradual", "sudden"],
        help="Distribution shift",
    )

    parser.add_argument("--model", nargs="+", required=True,
        choices=["resnet18", "resnet50", "densenet121", "inception_v3", "wide_resnet50", "googlenet",
                 "efficientnet_b0","mobilenet_v2"],
        help="models to run on.",
    )

    parser.add_argument("--lr", default=1e-3, help="Learning rate for training.")
    #NOTE: Change
    default_batch = 128 if torch.backends.mps.is_available() else 64
    parser.add_argument("--batch_size", default=default_batch, help="Batch size for data loader.")
    parser.add_argument("--n_epochs", default=120, help="Number of epochs to train for.")
    parser.add_argument("--patience", default=10, help="Number of epochs before early stopping.")
    parser.add_argument("--ignore_checkpoint", action="store_true", help="Whether to restart from scratch.")
    parser.add_argument("--target_cov", default=90, type=int, help="The target coverage (as a percent).")
    args = parser.parse_args()
    assert 50 < args.target_cov < 100
    args.target_cov = args.target_cov / 100

    # Set up distributed training if desired, and set the device
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if args.local_rank == -1:
        #if torch.backends.mps.is_available():
        #    args.device = torch.device("mps")
        #    print("Using Mac GPU (MPS)")
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            args.device = torch.device("cpu")
            print("Using CPU")
        args.world_size = 1
    else:
        dist.init_process_group(backend="nccl")
        args.device = torch.device(args.local_rank)
        args.world_size = dist.get_world_size()

    return args


def get_base_dataset(dataset, split):
    if dataset == "ImageNet":
        return ImageNet(split)
    elif dataset == "TinyImageNet":
        return TinyImageNet(split)
    elif dataset == "CIFAR10":
        return CIFAR10(split)
    elif dataset == "CIFAR100":
        return CIFAR100(split)
    raise ValueError(f"Dataset {dataset} is not supported.")


def get_model_file(args, curr_model):
    rootdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(rootdir, "cv_models", args.dataset, curr_model, "model.pt")


def get_model(args, curr_model):
    if args.dataset != "ImageNet":
        return torch.load(get_model_file(args, curr_model), map_location = args.device)
    return create_model(dataset=ImageNet("valid"), model_name = curr_model, device=args.device)


def get_results_file(args, curr_model, corruption, severity):
    rootdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(rootdir, "cv_logits", args.dataset, curr_model, f"{corruption}_{severity}.pt")


def get_temp_file(args, curr_model):
    return os.path.join(os.path.dirname(get_results_file(args, curr_model, None, 0)), "temp.txt")


def finished(args, curr_model):
    for corruption in corruptions:
        for severity in [0] if corruption is None else [1, 2, 3, 4, 5]:
            fname = get_results_file(args, curr_model, corruption, severity)
            if not os.path.isfile(fname):
                return False
    return os.path.isfile(get_temp_file(args, curr_model))


def raps_params(dataset):
    if dataset == "CIFAR10":
        lmbda, k_reg, n_class = 0.1, 1, 10
    elif dataset == "CIFAR100":
        lmbda, k_reg, n_class = 0.02, 5, 100
    elif dataset == "TinyImageNet":
        lmbda, k_reg, n_class = 0.01, 20, 200
    elif dataset == "ImageNet":
        lmbda, k_reg, n_class = 0.01, 10, 1000
    else:
        raise ValueError(f"Unsupported dataset {dataset}")
    return lmbda, k_reg, n_class


def train(args, curr_model):
    # Get train/valid data
    train_data = get_base_dataset(args.dataset, "train")
    valid_data = get_base_dataset(args.dataset, "valid")

    # Load model checkpoint one has been saved. Otherwise, initialize everything from scratch.
    model_file = get_model_file(args, curr_model)
    ckpt_name = os.path.join(os.path.dirname(model_file), "checkpoint.pt")
    if os.path.isfile(ckpt_name) and not args.ignore_checkpoint:
        model, opt, epoch, best_epoch, best_valid_acc = torch.load(ckpt_name, map_location=args.device)
    else:
        # create save directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(os.path.dirname(ckpt_name), exist_ok=True)
        model = create_model(dataset=train_data, model_name = curr_model, device=args.device)
        if "ImageNet" in args.dataset:
            opt = SGD(model.parameters(), lr=0.1, momentum=0.9)
        else:
            opt = Adam(model.parameters(), lr=args.lr)
        epoch, best_epoch, best_valid_acc = 0, 0, 0.0

    # Set up distributed data parallel if applicable
    writer = args.local_rank in [-1, 0]
    if args.local_rank != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device])

    for epoch in range(epoch, args.n_epochs):
        # Check early stopping condition
        if args.patience and epoch - best_epoch > args.patience:
            break

        # Main training loop
        train_loader = data_loader(dataset=train_data, batch_size=args.batch_size // args.world_size, epoch=epoch)
        for x, y in tqdm.tqdm(train_loader, desc=f"Train epoch {epoch+1:2}/{args.n_epochs}", disable=not writer):
            opt.zero_grad()
            pred = model(x.to(device=args.device))
            loss = F.cross_entropy(pred, y.to(device=args.device))
            loss.backward()
            opt.step()

        # Anneal learning rate by a factor of 10 every 7 epochs
        if (epoch + 1) % 7 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.1

        # Obtain accuracy on the validation dataset
        valid_acc = torch.zeros(2, device=args.device)
        valid_loader = data_loader(valid_data, batch_size=args.batch_size, epoch=epoch)
        with torch.no_grad():
            for x, y in tqdm.tqdm(valid_loader, desc=f"Valid epoch {epoch + 1:2}/{args.n_epochs}", disable=True):
                pred = model(x.to(device=args.device))
                valid_acc[0] += x.shape[0]
                valid_acc[1] += (pred.argmax(dim=-1) == y.to(device=args.device)).sum().item()

        # Reduce results from all parallel processes
        if args.local_rank != -1:
            dist.all_reduce(valid_acc)
        valid_acc = (valid_acc[1] / valid_acc[0]).item()

        # Save checkpoint & update best saved model
        if writer:
            print(f"Epoch {epoch + 1:2} valid acc: {valid_acc:.5f}")
            model_to_save = model.module if args.local_rank != -1 else model
            if valid_acc > best_valid_acc:
                best_epoch = epoch
                best_valid_acc = valid_acc
                torch.save(model_to_save, model_file)
            torch.save([model_to_save, opt, epoch + 1, best_epoch, best_valid_acc], ckpt_name)

        # Synchronize before starting next epoch
        if args.local_rank != -1:
            dist.barrier()


def temperature_scaling(args, curr_model):
    temp = nn.Parameter(torch.tensor(1.0, device=args.device))
    #NOTE: Change
    #opt = LBFGS([temp], lr=0.01, max_iter=500)
    opt = Adam([temp], lr = 0.01)
    loss_fn = nn.CrossEntropyLoss()

    n_epochs = 10
    valid_data = get_base_dataset(args.dataset, "valid")
    model = get_model(args, curr_model)
    for epoch in range(n_epochs):
        valid_loader = data_loader(valid_data, batch_size=args.batch_size, epoch=epoch)
        for x, y in tqdm.tqdm(valid_loader, desc=f"Calibration epoch {epoch + 1:2}/{n_epochs}", disable=False):
            with torch.no_grad():
                logits = model(x.to(device=args.device))

            def eval():
                opt.zero_grad()
                loss = loss_fn(logits / temp, y.to(device=args.device))
                loss.backward()
                return loss

            opt.step(eval)

    return temp.item()


def get_logits(args, curr_model):
    if args.dataset == "CIFAR10":
        dataset_cls = CIFAR10C
    elif args.dataset == "CIFAR100":
        dataset_cls = CIFAR100C
    elif args.dataset == "TinyImageNet":
        dataset_cls = TinyImageNetC
    elif args.dataset == "ImageNet":
        dataset_cls = ImageNetC
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")
    model = None
    for corruption in tqdm.tqdm(corruptions, desc="Corruptions", position=1):
        severities = [0] if corruption is None else [1, 2, 3, 4, 5]
        for severity in tqdm.tqdm(severities, desc="Severity Levels", position=2, leave=False):
            fname = get_results_file(args, curr_model, corruption, severity)
            if os.path.isfile(fname) and not args.ignore_checkpoint:
                continue
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            if model is None:
                model = get_model(args, curr_model)

            # Save the model's logits & labels for the whole dataset
            logits, labels = [], []
            dataset = dataset_cls(corruption=corruption, severity=severity)
            loader = data_loader(dataset, batch_size=args.batch_size)
            with torch.no_grad():
                for x, y in loader:
                    logits.append(model(x.to(device=args.device)).cpu())
                    labels.append(y.cpu())
            torch.save([torch.cat(logits), torch.cat(labels)], fname)


def t_to_sev(t, window, run_length=500, schedule=None):
    if t < window or schedule in [None, "None", "none"]:
        return 0
    t_base = t - window // 2
    if schedule == "gradual":
        k = (t_base // run_length) % 10
        return k if k <= 5 else 10 - k
    return 5 * ((t_base // run_length) % 2)




def main():

    args = parse_args()

    # Train the model, save its logits on all the corrupted test datasets, and do temperature scaling
    sev2results = defaultdict(defaultdict)
    results = []
    for curr_model in args.model:
        if not finished(args, curr_model) and args.dataset != "ImageNet":
            train(args, curr_model)

        #if not finished(args) and args.dataset != "ImageNet":
        #    train(args)
        if args.local_rank in [-1, 0]:
            temp_file = get_temp_file(args, curr_model)
            if not finished(args, curr_model):
                get_logits(args, curr_model)
                temp = temperature_scaling(args, curr_model)
                with open(temp_file, "w") as f:
                    f.write(str(temp))

            # Load the saved logits
            with open(temp_file) as f:
                temp = float(f.readline())
            n_data = None
            sev2results_m = defaultdict(list)
            for corruption in corruptions:
                severities = [0] if corruption is None else [1, 2, 3, 4, 5]
                for severity in severities:
                    try:
                        logits, labels = torch.load(get_results_file(args, curr_model, corruption, severity))
                    except:
                        continue
                    sev2results_m[severity].append(list(zip(F.softmax(logits / temp, dim=-1).numpy(), labels.numpy())))
                    n_data = len(labels) if n_data is None else min(n_data, len(labels))    
        sev2results[curr_model] = sev2results_m
    # Initialize conformal prediction methods, along with accumulators for results
    lmbda, k_reg, n_class = raps_params(args.dataset)
    D = 1 + lmbda * np.sqrt(n_class - k_reg)
    
    previous_methods = [ScaleFreeOGD]
    
    all_methods = [ScaleFreeOGD, SAOCP_UPD]


    #iplementing new method
    new_saocp = [SAOCP_UPD]
    shift = args.distribution_shift
    states = range(10)

    saocp_upd_RT = []
    s_opts_new_saocp_allstates, w_opts_new_saocp_allstates, coverages_new_saocp_allstates, s_hats_new_saocp_allstates, widths_new_saocp_allstates = {}, {}, {}, {}, {}
    width_one_saocp_iupd = []
    for st in states:
        start_time = time.time()
        s_opts_new_saocp , w_opts_new_saocp, coverages_new_saocp, s_hats_new_saocp, widths_new_saocp  = [], [], [], [], []     
        width_one = 0
        window = 100
        state = np.random.RandomState(st)
        order = state.permutation(n_data)[: 8550]
        predictor_new_saocp  = new_saocp[0](None, None, max_scale=D, n_model=len(args.model), lifetime=8, coverage=args.target_cov)
        for t, i in tqdm.tqdm(enumerate(order, start=0), total=len(order)):
            sev = t_to_sev(t, window=window, schedule=shift)
            s_opt_all_model = []
            s_sort_cumsum = {mod_sum: None for mod_sum in args.model}
            single_s_opt_not_cumsum = []
            w_opt_all_model = []
            for m in args.model:
                sev2results_m = sev2results[m]
                probs , label = sev2results_m[sev][state.randint(0, len(sev2results_m[sev]))][i]

                # Convert probability to score
                i_sort = np.flip(np.argsort(probs))
                p_sort_cumsum = np.cumsum(probs[i_sort]) - state.rand() * probs[i_sort]
                s_sort_cumsum_m = p_sort_cumsum + lmbda * np.sqrt(np.cumsum([i > k_reg for i in range(n_class)]))
                s_sort_cumsum[m] = s_sort_cumsum_m

                w_opt = np.argsort(i_sort)[label] + 1
                s_opt = s_sort_cumsum_m[w_opt - 1]
                s_opt_all_model.append(s_opt)
                w_opt_all_model.append(w_opt)


            # Predict and Update conformal predictors
            name = type(predictor_new_saocp).__name__

            _, s_hat , model_prob = predictor_new_saocp.predict(horizon=1)
            s_sort_cumsum_l = list(s_sort_cumsum.values())
            s_sort_cumsum_l_w = list(model_prob.values())
            np.random.seed(st)
            selected_model = np.random.choice(len(model_prob.keys()), p=s_sort_cumsum_l_w)
            s_opts_new_saocp.append(s_opt_all_model[selected_model])
            w = np.sum(s_sort_cumsum_l[selected_model] <= s_hat)

            if w_opt_all_model[selected_model] <= w and w == 1:
                width_one+=1

            s_hats_new_saocp.append(s_hat)
            widths_new_saocp.append(w)

            coverages_new_saocp.append(w >= w_opt_all_model[selected_model])
            predictor_new_saocp.update(ground_truth=pd.Series(s_opt_all_model), forecast=pd.Series([0]*len(args.model)), horizon=1)
        
        end_time = time.time()
        runtime = end_time - start_time
        saocp_upd_RT.append(runtime)

        s_opts_new_saocp_allstates[st] = s_opts_new_saocp
        w_opts_new_saocp_allstates[st] = w_opts_new_saocp
        coverages_new_saocp_allstates[st] = coverages_new_saocp
        s_hats_new_saocp_allstates[st] = s_hats_new_saocp
        widths_new_saocp_allstates[st] = widths_new_saocp
        width_one_saocp_iupd.append(width_one)

    #end of implementing updated saocp

    for curr_model in args.model:
        sev2results_m = sev2results[curr_model]
        coverages_all_pm, s_hats_all_pm, widths_all_pm, single_width_pm = [{m.__name__: {} for m in previous_methods} for _ in range(4)]
        s_opt_all_pm, w_opt_all_pm = {} , {}
        prev_method_run_time = []
        for st in states:
            state = np.random.RandomState(st)    
            start_time = time.time()     
            s_opts, w_opts =  [], []
            window = 100

            single_width_pmethod = {m.__name__: 0 for m in previous_methods}
            
            order = state.permutation(n_data)[: 8550]

            coverages, s_hats, widths = [{m.__name__: [] for m in previous_methods} for _ in range(3)]
            predictors = [m(None, None, max_scale=D, lifetime=32, coverage=args.target_cov, rs=st) for m in previous_methods]
            for t, i in tqdm.tqdm(enumerate(order, start=0), total=len(order)):  
                # Get saved results for the desired severity
                sev = t_to_sev(t, window=window, schedule=shift)               
                probs, label = sev2results_m[sev][state.randint(0, len(sev2results_m[sev]))][i]

                # Convert probability to score
                i_sort = np.flip(np.argsort(probs))
                p_sort_cumsum = np.cumsum(probs[i_sort]) - state.rand() * probs[i_sort]
                s_sort_cumsum = p_sort_cumsum + lmbda * np.sqrt(np.cumsum([i > k_reg for i in range(n_class)]))
                w_opt = np.argsort(i_sort)[label] + 1
                s_opt = s_sort_cumsum[w_opt - 1]
                s_opts.append(s_opt)
                w_opts.append(w_opt)


                # Update all the conformal predictors
                for predictor in predictors:
                    name = type(predictor).__name__

                    _, s_hat = predictor.predict(horizon=1)

                    w = np.sum(s_sort_cumsum <= s_hat)

                    if w >= w_opt and w == 1:
                        single_width_pmethod[name]+=1


                    s_hats[name].append(s_hat)
                    widths[name].append(w)
                    coverages[name].append(w >= w_opt)
                    predictor.update(ground_truth=pd.Series([s_opt]), forecast=pd.Series([0]), horizon=1)


            
            s_opt_all_pm[st] = s_opts
            w_opt_all_pm[st] = w_opts
            for predictor in predictors:
                pv_method = type(predictor).__name__
                coverages_all_pm[pv_method][st] = coverages[pv_method]
                s_hats_all_pm[pv_method][st] = s_hats[pv_method]
                widths_all_pm[pv_method][st] = widths[pv_method]
                single_width_pm[pv_method][st] = single_width_pmethod[pv_method]
        

            end_time = time.time()
            runtime = end_time - start_time
            prev_method_run_time.append(runtime)


        print(f"Distribution shift: {shift}")
        for i, m in enumerate(all_methods):
            # Compute various summary statistics
            AVG_reg = []
            all_state_coverage = []
            all_state_width = []
            runtime = []
            single_width = []

            if m == SAOCP_UPD:
                name = m.__name__

                for s in states:
                    s_opts_n = np.asarray(s_opts_new_saocp_allstates[s])
                    int_q = pd.Series(s_opts_n).rolling(window).quantile(args.target_cov).dropna()
                    s_hat = np.asarray(s_hats_new_saocp_allstates[s])
                    int_losses = pd.Series(pinball_loss(s_opts_new_saocp_allstates[s], s_hat, args.target_cov)).rolling(window).mean().dropna()
                    opts = [pinball_loss(s_opts_n[i : i + window], q, args.target_cov).mean() for i, q in enumerate(int_q)]
                    int_regret = int_losses.values - np.asarray(opts)


                    all_state_coverage.append(np.mean(coverages_new_saocp_allstates[s]))
                    all_state_width.append(np.mean(widths_new_saocp_allstates[s]))
                    AVG_reg.append(np.mean(int_regret))
                    runtime = saocp_upd_RT
                    single_width.append(width_one_saocp_iupd[s])
                    


            else:    
                name = m.__name__
                for st in states:
                    s_opts = np.array(s_opt_all_pm[st])
                    int_q = pd.Series(s_opt_all_pm[st]).rolling(window).quantile(args.target_cov).dropna()
                    s_hat = np.asarray(s_hats_all_pm[name][st])
                    int_losses = pd.Series(pinball_loss(s_opts, s_hat, args.target_cov)).rolling(window).mean().dropna()
                    opts = [pinball_loss(s_opts[i : i + window], q, args.target_cov).mean() for i, q in enumerate(int_q)]
                    int_regret = int_losses.values - np.asarray(opts)
                    all_state_coverage.append(np.mean(coverages_all_pm[name][st]))
                    all_state_width.append(np.mean(widths_all_pm[name][st]))
                    AVG_reg.append(np.mean(int_regret))
                    runtime = prev_method_run_time
                    single_width.append(single_width_pm[name][st])


            

            row = {'Shift': shift,
                    'Model' :curr_model,
                    'Method':name,
                    'Cov': f"{np.mean(all_state_coverage):.4f} \u00B1 {np.std(all_state_coverage):.4f}",
                    'Avg Width': f"{np.mean(all_state_width):.4f} \u00B1 {np.std(all_state_width):.4f}",
                    'Avg Regret': f"{np.mean(AVG_reg):.10f} \u00B1 {np.std(AVG_reg):.10f}",
                    'Run Time': f"{np.mean(runtime):.4f} \u00B1 {np.std(runtime):.4f}",
                    'Single Width':f"{np.mean(single_width):.10f} \u00B1 {np.std(single_width):.10f}"
                    }
            results.append(row)
            
            print(

                f"{curr_model:10}"
                f"{name:10}"
                f"Cov: {np.mean(all_state_coverage):.4f} \u00B1 {np.std(all_state_coverage):.4f}"
                f"Avg Width: {np.mean(all_state_width):.2f} \u00B1 {np.std(all_state_width):.2f}"
                f"Avg Regret: {np.mean(AVG_reg):.7f} \u00B1 {np.std(AVG_reg):.7f}"
                f"Run Time: {np.mean(runtime):.2f} \u00B1 {np.std(runtime):.2f}"
                f"Single Width:{np.mean(single_width):.5f} \u00B1 {np.std(single_width):.5f}"

            )
                
                


    csv_file = "outputsum.csv"

    # CSV fieldnames
    fieldnames = ['Shift', 'Model','Method' , 'Cov', 'Avg Width', 'Avg Regret','Run Time', 'Single Width']

    # Writing to CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header
        writer.writeheader()            
            # Write data rows
        for row in results:
            writer.writerow(row)
                    



if __name__ == "__main__":
    main()
