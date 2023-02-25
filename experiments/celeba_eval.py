# future
from __future__ import absolute_import, division, print_function

# stdlib
import argparse
import os
from typing import Dict

# third party
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn import metrics

# domias absolute
from domias.baselines import baselines, compute_metrics_baseline
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
from domias.metrics.wd import compute_wd
from domias.models.ctgan import CTGAN

PATH_CELEB_REPRESENTATION = "celebA_representation"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--density_estimator", type=str, default="bnaf", choices=["bnaf", "kde"]
)
parser.add_argument(
    "--mem_set_size_list",
    nargs="+",
    type=int,
    default=[100],  # 999
    help="size of training dataset",
)
parser.add_argument(
    "--reference_set_size_list",
    nargs="+",
    type=int,
    default=[100],  # 4500
    help="size of held-out dataset",
)
parser.add_argument(
    "--training_epoch_list",
    nargs="+",
    type=int,
    default=[100],  # 2000
    help="# training epochs",
)
parser.add_argument(
    "--synthetic_sizes",
    nargs="+",
    type=int,
    default=[100],  # 50000
    help="size of generated dataset",
)
parser.add_argument("--device", type=str, default=None)

parser.add_argument("--gpu_idx", default=None, type=int)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--rep_dim", type=int, default=128)
parser.add_argument("--mem_set_size", type=int, default=4000)


args = parser.parse_args()
args.device = f"cuda:{args.gpu_idx}"

GPU_IDX = args.gpu_idx
SEED = args.seed
LATENT_REPRESENTATION_DIM = args.rep_dim

alias = f"{SEED}_{GPU_IDX}_{LATENT_REPRESENTATION_DIM}_tsz{args.mem_set_size}"


if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results_folder", exist_ok=True)


performance_logger: Dict = {}

""" 1. load dataset"""
mem_set = np.load(
    f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_real_{alias}.npy"
)
non_mem_set = np.load(
    f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_test_{alias}.npy"
)[: 999 + args.mem_set_size - 1000]
reference_set = np.load(
    f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_ref_{alias}.npy"
)[:4500]

""" 2. training-test-addition split"""
for mem_set_size in args.mem_set_size_list:
    for reference_set_size in args.reference_set_size_list:
        for training_epochs in args.training_epoch_list:
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ] = {}
            for N_DATA_GEN in args.synthetic_sizes:
                samples = pd.DataFrame(
                    np.load(
                        f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_synth_{alias}.npy"
                    )[:N_DATA_GEN]
                )
                samples_val = pd.DataFrame(
                    np.load(
                        f"{PATH_CELEB_REPRESENTATION}/AISTATS_betavae_repres_synth_{alias}.npy"
                    )[N_DATA_GEN : N_DATA_GEN * 2]
                )
            wd_n = min(len(samples), len(reference_set))
            eval_met_on_held_out = compute_wd(samples[:wd_n], reference_set[:wd_n])
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_evaluation"] = eval_met_on_held_out
            print(
                "SIZE: ",
                mem_set_size,
                "TVAE EPOCH: ",
                training_epochs,
                "N_DATA_GEN: ",
                N_DATA_GEN,
                "reference_set_size: ",
                reference_set_size,
                "Performance (Sample-Quality): ",
                eval_met_on_held_out,
            )
            X_test = np.concatenate([mem_set, non_mem_set])
            Y_test = np.concatenate(
                [np.ones(mem_set.shape[0]), np.zeros(non_mem_set.shape[0])]
            ).astype(bool)

            # build another GAN for LOGAN 0

            """ 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)"""
            if args.density_estimator == "bnaf":
                _gen, model_gen = density_estimator_trainer(
                    samples.values,
                    samples_val.values[: int(0.5 * N_DATA_GEN)],
                    samples_val.values[int(0.5 * N_DATA_GEN) :],
                )
                _data, model_data = density_estimator_trainer(
                    reference_set,
                    reference_set[: int(0.5 * reference_set_size)],
                    reference_set[: int(0.5 * reference_set_size)],
                )
                p_G_evaluated = np.exp(
                    compute_log_p_x(
                        model_gen, torch.as_tensor(X_test).float().to(device)
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

            elif args.density_estimator == "kde":
                density_gen = stats.gaussian_kde(samples.values.transpose(1, 0))
                density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
                p_G_evaluated = density_gen(X_test.transpose(1, 0))

            ctgan = CTGAN(epochs=200)
            samples.columns = [str(_) for _ in range(mem_set.shape[1])]
            ctgan.fit(samples)  # train a CTGAN on the generated examples

            if ctgan._transformer is None or ctgan._discriminator is None:
                raise RuntimeError()

            ctgan_representation = ctgan._transformer.transform(X_test)
            ctgan_score = (
                ctgan._discriminator(
                    torch.as_tensor(ctgan_representation).float().to(device)
                )
                .cpu()
                .detach()
                .numpy()
            )

            acc, auc = compute_metrics_baseline(ctgan_score, Y_test)

            baseline_results, baseline_scores = baselines(
                X_test,
                Y_test,
                samples.values,
                reference_set,
                reference_set,
            )
            baseline_results["LOGAN_0"] = {"accuracy": acc, "aucroc": auc}
            baseline_scores["LOGAN_0"] = ctgan_score
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_Baselines"] = baseline_results
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_BaselineScore"] = baseline_scores
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_Xtest"] = X_test
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_Ytest"] = Y_test

            #                 eqn1: \prop P_G(x_i)
            log_p_test = p_G_evaluated
            thres = np.quantile(log_p_test, 0.5)
            auc_y = np.hstack(
                (
                    np.array([1] * mem_set.shape[0]),
                    np.array([0] * non_mem_set.shape[0]),
                )
            )
            fpr, tpr, thresholds = metrics.roc_curve(auc_y, log_p_test, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            print(
                "Eqn.(1), training set prediction acc",
                (p_G_evaluated > thres).sum(0) / mem_set_size,
            )
            print("Eqn.(1), AUC", auc)
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_Eqn1"] = (p_G_evaluated > thres).sum(0) / mem_set_size
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_Eqn1AUC"] = auc
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_Eqn1Score"] = log_p_test
            # eqn2: \prop P_G(x_i)/P_X(x_i)
            if args.density_estimator == "bnaf":
                p_R_evaluated = np.exp(
                    compute_log_p_x(
                        model_data, torch.as_tensor(X_test).float().to(device)
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

            elif args.density_estimator == "kde":
                p_R_evaluated = density_data(X_test.transpose(1, 0)) + 1e-30

            p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

            acc, auc = compute_metrics_baseline(p_rel, Y_test)
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_Eqn2"] = acc

            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_Eqn2AUC"] = auc
            performance_logger[
                f"{mem_set_size}_{training_epochs}_{reference_set_size}"
            ][f"{N_DATA_GEN}_Eqn2Score"] = p_rel

            print(performance_logger)
