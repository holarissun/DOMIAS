# future
from __future__ import absolute_import, division, print_function

# stdlib
from typing import Any, Dict, Optional

# third party
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import multivariate_normal

# domias absolute
from domias.baselines import baselines, compute_metrics_baseline
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
from domias.metrics.wd import compute_wd
from domias.models.ctgan import CTGAN
from domias.models.generator import GeneratorInterface

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class gaussian:
    def __init__(self, X: np.ndarray) -> None:
        var = np.std(X, axis=0) ** 2
        mean = np.mean(X, axis=0)
        self.rv = multivariate_normal(mean, np.diag(var))

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return self.rv.pdf(Z)


class normal_func:
    def __init__(self, X: np.ndarray) -> None:
        self.var = np.ones_like(np.std(X, axis=0) ** 2)
        self.mean = np.zeros_like(np.mean(X, axis=0))

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(Z, self.mean, np.diag(self.var))
        # return multivariate_normal.pdf(Z, np.zeros_like(self.mean), np.diag(np.ones_like(self.var)))


class normal_func_feat:
    def __init__(
        self,
        X: np.ndarray,
        continuous: list,
    ) -> None:
        if np.any(np.array(continuous) > 1) or len(continuous) != X.shape[1]:
            raise ValueError("Continous variable needs to be boolean")
        self.feat = np.array(continuous).astype(bool)

        if np.sum(self.feat) == 0:
            raise ValueError("there needs to be at least one continuous feature")

        for i in np.arange(X.shape[1])[self.feat]:
            if len(np.unique(X[:, i])) < 10:
                print(f"Warning: feature {i} does not seem continous. CHECK")

        self.var = np.std(X[:, self.feat], axis=0) ** 2
        self.mean = np.mean(X[:, self.feat], axis=0)

    def pdf(self, Z: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(Z[:, self.feat], self.mean, np.diag(self.var))


def evaluate_performance(
    generator: GeneratorInterface,
    dataset: np.ndarray,
    mem_set_size: int,
    reference_set_size: int,
    training_epochs: int = 2000,
    synthetic_sizes: list = [10000],
    density_estimator: str = "prior",
    seed: int = 0,
    device: Any = DEVICE,
    shifted_column: Optional[int] = None,
    zero_quantile: float = 0.3,
    reference_kept_p: float = 1.0,
) -> Dict:
    """
    Evaluate various Membership Inference Attacks, using the `generator` and the `dataset`.
    The provided generator must not be fitted.

    Args:
        generator: GeneratorInterface
            Generator with the `fit` and `generate` methods. The generator MUST not be fitted.
        dataset: int
            The evaluation dataset, used to derive the training and test datasets.
        training_size: int
            The split for the training (member) dataset out of `dataset`
        reference_size: int
            The split for the reference dataset out of `dataset`.
        training_epochs: int
            Training epochs
        synthetic_sizes: List[int]
            For how many synthetic samples to test the attacks.
        density_estimator: str, default = "prior"
            Which density to use. Available options:
                * prior
                * bnaf
                * kde
        seed: int
            Random seed
        device: PyTorch device
            CPU or CUDA
        shifted_column: Optional[int]
            Shift a column
        zero_quantile: float
            Threshold for shifting the column.
        reference_kept_p: float
            Reference dataset parameter (for distributional shift experiment)

    Returns:
        A dictionary with a key for each of the `synthetic_sizes` values.
        For each `synthetic_sizes` value, the dictionary contains the keys:
            * `MIA_performance` : accuracy and AUCROC for each attack
            * `MIA_scores`: output scores for each attack
            * `data`: the evaluation data
        For both `MIA_performance` and `MIA_scores`, the following attacks are evaluated:
            * "ablated_eq1"
            * "ablated_eq2"
            * "LOGAN_D1"
            * "MC"
            * "gan_leaks"
            * "gan_leaks_cal"
            * "LOGAN_0"
            * "eq1"
            * "domias"
    """
    performance_logger: Dict = {}

    continuous = []
    for i in np.arange(dataset.shape[1]):
        if len(np.unique(dataset[:, i])) < 10:
            continuous.append(0)
        else:
            continuous.append(1)

    norm = normal_func_feat(dataset, continuous)

    # For experiment with domain shift in reference dataset
    if shifted_column is not None:
        thres = np.quantile(dataset[:, shifted_column], zero_quantile) + 0.01
        dataset[:, shifted_column][dataset[:, shifted_column] < thres] = -999.0
        dataset[:, shifted_column][dataset[:, shifted_column] > thres] = 999.0
        dataset[:, shifted_column][dataset[:, shifted_column] == -999.0] = 0.0
        dataset[:, shifted_column][dataset[:, shifted_column] == 999.0] = 1.0

        mem_set = dataset[:mem_set_size]  # membership set
        mem_set = mem_set[mem_set[:, shifted_column] == 1]

        non_mem_set = dataset[mem_set_size : 2 * mem_set_size]  # set of non-members
        non_mem_set = non_mem_set[: len(mem_set)]
        reference_set = dataset[-reference_set_size:]

        # Used for experiment with distributional shift in reference dataset
        reference_set_A1 = reference_set[reference_set[:, shifted_column] == 1]
        reference_set_A0 = reference_set[reference_set[:, shifted_column] == 0]
        reference_set_A0_kept = reference_set_A0[
            : int(len(reference_set_A0) * reference_kept_p)
        ]
        if reference_kept_p > 0:
            reference_set = np.concatenate((reference_set_A1, reference_set_A0_kept), 0)
        else:
            reference_set = reference_set_A1
            # non_mem_set = non_mem_set_A1

        mem_set_size = len(mem_set)
        reference_set_size = len(reference_set)

        # hide column A
        mem_set = np.delete(mem_set, shifted_column, 1)
        non_mem_set = np.delete(non_mem_set, shifted_column, 1)
        reference_set = np.delete(reference_set, shifted_column, 1)
        dataset = np.delete(dataset, shifted_column, 1)
    # For all other experiments
    else:
        mem_set = dataset[:mem_set_size]
        non_mem_set = dataset[mem_set_size : 2 * mem_set_size]
        reference_set = dataset[-reference_set_size:]

    """ 3. Synthesis with the GeneratorInferface"""
    df = pd.DataFrame(mem_set)
    df.columns = [str(_) for _ in range(dataset.shape[1])]

    # Train generator
    generator.fit(df)

    for synthetic_size in synthetic_sizes:
        performance_logger[synthetic_size] = {
            "MIA_performance": {},
            "MIA_scores": {},
            "data": {},
        }
        synth_set = generator.generate(synthetic_size)
        synth_val_set = generator.generate(synthetic_size)

        wd_n = min(len(synth_set), len(reference_set))
        eval_met_on_reference = compute_wd(synth_set[:wd_n], reference_set[:wd_n])
        performance_logger[synthetic_size]["MIA_performance"][
            "sample_quality"
        ] = eval_met_on_reference

        # get real test sets of members and non members
        X_test = np.concatenate([mem_set, non_mem_set])
        Y_test = np.concatenate(
            [np.ones(mem_set.shape[0]), np.zeros(non_mem_set.shape[0])]
        ).astype(bool)

        performance_logger[synthetic_size]["data"]["Xtest"] = X_test
        performance_logger[synthetic_size]["data"]["Ytest"] = Y_test

        """ 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)"""
        # First, estimate density of synthetic data
        # BNAF for pG
        if density_estimator == "bnaf":
            _, p_G_model = density_estimator_trainer(
                synth_set.values,
                synth_val_set.values[: int(0.5 * synthetic_size)],
                synth_val_set.values[int(0.5 * synthetic_size) :],
            )
            _, p_R_model = density_estimator_trainer(reference_set)
            p_G_evaluated = np.exp(
                compute_log_p_x(p_G_model, torch.as_tensor(X_test).float().to(device))
                .cpu()
                .detach()
                .numpy()
            )

        # KDE for pG
        elif density_estimator == "kde":
            density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
            density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
            p_G_evaluated = density_gen(X_test.transpose(1, 0))
        elif density_estimator == "prior":
            density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
            density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
            p_G_evaluated = density_gen(X_test.transpose(1, 0))

        # Baselines
        baseline_results, baseline_scores = baselines(
            X_test,
            Y_test,
            synth_set.values,
            reference_set,
            reference_set,  # we pass the reference dataset to GAN-leaks CAL for better stability and fairer comparison (compared to training additional model, as Chen et al propose).
        )

        performance_logger[synthetic_size]["MIA_performance"] = baseline_results
        performance_logger[synthetic_size]["MIA_scores"] = baseline_scores

        # build another GAN for LOGAN 0 black-box
        ctgan = CTGAN(epochs=training_epochs, pac=1)
        synth_set.columns = [str(_) for _ in range(dataset.shape[1])]
        ctgan.fit(synth_set)  # train a CTGAN on the generated examples

        if ctgan._transformer is None or ctgan._discriminator is None:
            raise RuntimeError()
        # add LOGAN 0 baseline
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

        performance_logger[synthetic_size]["MIA_performance"]["LOGAN_0"] = {
            "accuracy": acc,
            "aucroc": auc,
        }
        performance_logger[synthetic_size]["MIA_scores"]["LOGAN_0"] = ctgan_score

        # Ablated version, based on eqn1: \prop P_G(x_i)
        acc, auc = compute_metrics_baseline(p_G_evaluated, Y_test)

        performance_logger[synthetic_size]["MIA_performance"]["eq1"] = {
            "accuracy": acc,
            "aucroc": auc,
        }
        performance_logger[synthetic_size]["MIA_scores"]["eq1"] = p_G_evaluated

        # eqn2: \prop P_G(x_i)/P_X(x_i)
        # DOMIAS (BNAF for p_R estimation)
        if density_estimator == "bnaf":
            p_R_evaluated = np.exp(
                compute_log_p_x(p_R_model, torch.as_tensor(X_test).float().to(device))
                .cpu()
                .detach()
                .numpy()
            )

        # DOMIAS (KDE for p_R estimation)
        elif density_estimator == "kde":
            p_R_evaluated = density_data(X_test.transpose(1, 0))

        # DOMIAS (with prior for p_R, see Appendix experiment)
        elif density_estimator == "prior":
            p_R_evaluated = norm.pdf(X_test)

        p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

        acc, auc = compute_metrics_baseline(p_rel, Y_test)
        performance_logger[synthetic_size]["MIA_performance"]["domias"] = {
            "accuracy": acc,
            "aucroc": auc,
        }

        performance_logger[synthetic_size]["MIA_scores"]["domias"] = p_rel

    return performance_logger
