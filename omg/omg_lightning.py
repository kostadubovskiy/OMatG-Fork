from enum import Enum, auto
from pathlib import Path
import time
from typing import Dict, Optional, Sequence
from ase import Atoms
import lightning
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from omg.analysis import get_coordination_numbers, get_cov, match_rmsds, ValidAtoms
from omg.datamodule import OMGData
from omg.globals import SMALL_TIME, BIG_TIME
from omg.model.model import Model
from omg.sampler.minimum_permutation_distance import correct_for_minimum_permutation_distance
from omg.sampler.sampler import Sampler
from omg.si.abstracts import StochasticInterpolantSpecies
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.utils import xyz_saver


class OMGLightning(lightning.LightningModule):
    """
    Main module which is fit and used to generate structures using Lightning CLI.
    """

    class ValidationMetric(Enum):
        """
        Enum for the possible types of reported validation metrics.
        """

        LOSS = auto()
        """
        Ordinary loss.
        """
        MATCH_RATE = auto()
        """
        Match rate for the CSP task.
        """
        DNG_EVAL = auto()
        """
        Evaluation for the DNG task.
        """

    def __init__(self, si: StochasticInterpolants, sampler: Sampler, model: Model,
                 relative_si_costs: Dict[str, float], use_min_perm_dist: bool = False,
                 generation_xyz_filename: Optional[str] = None, sobol_time: bool = False,
                 float_32_matmul_precision: str = "medium", validation_mode: str = "loss",
                 dataset_name: str = "mp_20", number_cpus: int = 12) -> None:
        super().__init__()
        self.si = si
        self.sampler = sampler
        self.use_min_perm_dist = use_min_perm_dist
        if self.use_min_perm_dist:
            self._pos_corrector = self.si.get_stochastic_interpolant("pos").get_corrector()
        else:
            self._pos_corrector = None
        species_stochastic_interpolant = self.si.get_stochastic_interpolant("species")
        if not isinstance(species_stochastic_interpolant, StochasticInterpolantSpecies):
            raise ValueError("Species stochastic interpolant must be of type StochasticInterpolantSpecies.")
        if species_stochastic_interpolant.uses_masked_species():
            model.enable_masked_species()
        self.model = model

        if not all(cost >= 0.0 for cost in relative_si_costs.values()):
            raise ValueError("All cost factors must be non-negative.")
        if not abs(sum(cost for cost in relative_si_costs.values()) - 1.0) < 1e-10:
            raise ValueError("The sum of all cost factors should be equal to 1.")
        si_loss_keys = self.si.loss_keys()
        for key in relative_si_costs.keys():
            if key not in si_loss_keys:
                raise ValueError(f"Loss key {key} not found in the stochastic interpolants.")
        for key in si_loss_keys:
            if key not in relative_si_costs.keys():
                raise ValueError(f"Loss key {key} not found in the costs.")
        self._relative_si_costs = relative_si_costs

        if not sobol_time:
            # Don't sample between 0 and 1 because the gamma function may diverge as t -> 0 or t -> 1, which
            # may result in NaN losses during training if t was to close to 0 or 1 (especially at 32-true precision).
            self.time_sampler = lambda n: torch.rand(n) * (BIG_TIME - SMALL_TIME) + SMALL_TIME
        else:
            # Don't sample between 0 and 1 because the gamma function may diverge as t -> 0 or t -> 1, which
            # may result in NaN losses during training if t was to close to 0 or 1 (especially at 32-true precision).
            self.time_sampler = (
                lambda n: torch.reshape(torch.quasirandom.SobolEngine(dimension=1, scramble=True).draw(n), (-1,))
                          * (BIG_TIME - SMALL_TIME) + SMALL_TIME)
        self.generation_xyz_filename = generation_xyz_filename

        try:
            self._validation_metric = self.ValidationMetric[validation_mode.upper()]
        except AttributeError:
            raise ValueError(f"Unknown validation metric f{validation_mode}.")
        self.dataset_name = dataset_name
        if not self.dataset_name in ["mp_20", "carbon_24", "perov_5", "mpts_52", "alex_mp_20"]:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")
        self.number_cpus = number_cpus
        if not self.number_cpus >= 1:
            raise ValueError(f"Number of CPUs {self.number_cpus} must be >= 1.")

        # Possible values are "medium", "high", and "highest".
        torch.set_float32_matmul_precision(float_32_matmul_precision)

        self.reference_atoms = []
        self.generated_atoms = []

    def forward(self, x_t: Sequence[torch.Tensor], t: torch.Tensor) -> Sequence[Sequence[torch.Tensor]]:
        """
        Calls encoder + head stack

        :param x_t:
            Sequence of torch.tensors corresponding to batched species, fractional coordinates and lattices.
        :type x_t: Sequence[torch.Tensor]

        :param t:
            Sampled times
        :type t: torch.Tensor

        :return:
            Predicted b and etas for species, coordinates and lattices, respectively.
        :rtype: Sequence[Sequence[torch.Tensor]]
        """
        x = self.model(x_t, t)
        return x

    def training_step(self, x_1: OMGData) -> torch.Tensor:
        """
        Performs one training step given a batch of x_1

        :return:
            Loss from training step
        :rtype: torch.Tensor
        """
        x_0 = self.sampler.sample_p_0(x_1).to(self.device)

        # Minimize permutational distance between clusters.
        if self.use_min_perm_dist:
            # Don't switch species to allow for crystal-structure prediction.
            correct_for_minimum_permutation_distance(x_0, x_1, self._pos_corrector, switch_species=False)

        # Sample t for each structure.
        t = self.time_sampler(len(x_1.n_atoms)).to(self.device)

        losses = self.si.losses(self.model, t, x_0, x_1)

        total_loss = torch.tensor(0.0, device=self.device)

        for loss_key in losses:
            losses[loss_key] = self._relative_si_costs[loss_key] * losses[loss_key]
            total_loss += losses[loss_key]

        assert "loss_total" not in losses
        losses["loss_total"] = total_loss

        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(x_1.n_atoms)
        )

        return total_loss

    def on_validation_epoch_start(self):
        self.reference_atoms = []
        self.generated_atoms = []

    def validation_step(self, x_1: OMGData) -> torch.Tensor:
        batch_size = len(x_1.n_atoms)
        x_0 = self.sampler.sample_p_0(x_1).to(self.device)

        if (self._validation_metric == self.ValidationMetric.MATCH_RATE
                or self._validation_metric == self.ValidationMetric.DNG_EVAL):
            # Prevent moving x_1 to cpu because it's needed below.
            x_1_cpu = x_1.clone().to('cpu')
            for i in range(batch_size):
                lower, upper = x_1_cpu.ptr[i], x_1_cpu.ptr[i + 1]
                self.reference_atoms.append(
                    Atoms(numbers=x_1_cpu.species[lower:upper], scaled_positions=x_1_cpu.pos[lower:upper, :],
                          cell=x_1_cpu.cell[i, :, :], pbc=(1, 1, 1)))

            gen = self.si.integrate(x_0, self.model, save_intermediate=False)
            gen.to('cpu')
            assert len(gen.n_atoms) == batch_size
            assert torch.equal(gen.n_atoms, x_1_cpu.n_atoms)
            assert torch.equal(gen.ptr, x_1_cpu.ptr)
            for i in range(batch_size):
                lower, upper = gen.ptr[i], gen.ptr[i + 1]
                self.generated_atoms.append(
                    Atoms(numbers=gen.species[lower:upper], scaled_positions=gen.pos[lower:upper, :],
                          cell=gen.cell[i, :, :], pbc=(1, 1, 1)))
        else:
            assert self._validation_metric == self.ValidationMetric.LOSS

        # Minimize permutational distance between clusters. Should be done after integrating.
        if self.use_min_perm_dist:
            # Don't switch species to allow for crystal-structure prediction.
            correct_for_minimum_permutation_distance(x_0, x_1, self._pos_corrector, switch_species=False)

        # Sample t for each structure.
        t = self.time_sampler(len(x_1.n_atoms)).to(self.device)

        losses = self.si.losses(self.model, t, x_0, x_1)

        total_loss = torch.tensor(0.0, device=self.device)

        # Force creation of copy of keys because dictionary will be changed in iteration.
        for loss_key in list(losses):
            losses[f"val_{loss_key}"] = self._relative_si_costs[loss_key] * losses[loss_key]
            total_loss += losses[f"val_{loss_key}"]
            losses.pop(loss_key)

        assert "loss_total" not in losses
        losses["val_loss_total"] = total_loss

        self.log_dict(
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )

        return total_loss

    def on_validation_epoch_end(self):
        if self._validation_metric == self.ValidationMetric.MATCH_RATE:
            assert len(self.generated_atoms) == len(self.reference_atoms)
            gen_valid_atoms = ValidAtoms.get_valid_atoms(self.generated_atoms, desc="Validating generated structures",
                                                         skip_validation=True, number_cpus=1)
            ref_valid_atoms = ValidAtoms.get_valid_atoms(self.reference_atoms, desc="Validating reference structures",
                                                         skip_validation=True, number_cpus=1)

            rmsds, valid_rmsds = match_rmsds(gen_valid_atoms, ref_valid_atoms, ltol=0.3, stol=0.5, angle_tol=10.0,
                                             number_cpus=self.number_cpus, enable_progress_bar=True)
            match_count = sum(rmsd is not None for rmsd in rmsds)
            match_rate = match_count / len(gen_valid_atoms)
            filtered_rmsds = [rmsd for rmsd in rmsds if rmsd is not None]
            mean_rmsd = np.mean(filtered_rmsds)

            self.log("match_rate", match_rate, sync_dist=True)
            self.log("mean_rmsd", float(mean_rmsd), sync_dist=True)
        elif self._validation_metric == self.ValidationMetric.DNG_EVAL:
            assert len(self.generated_atoms) == len(self.reference_atoms)
            gen_valid_atoms = ValidAtoms.get_valid_atoms(self.generated_atoms, desc="Validating generated structures",
                                                         number_cpus=self.number_cpus)
            ref_valid_atoms = ValidAtoms.get_valid_atoms(self.reference_atoms, desc="Validating reference structures",
                                                         number_cpus=self.number_cpus)
            valid_rate = sum(va.valid for va in gen_valid_atoms) / len(gen_valid_atoms)

            gen_densities = [struc.structure.density for struc in gen_valid_atoms]
            ref_densities = [struc.structure.density for struc in ref_valid_atoms]
            wdist_density = wasserstein_distance(gen_densities, ref_densities)
            gen_narity = [len(set(struc.structure.species)) for struc in gen_valid_atoms]
            ref_narity = [len(set(struc.structure.species)) for struc in ref_valid_atoms]
            wdist_narity = wasserstein_distance(gen_narity, ref_narity)
            gen_coordination_numbers = [np.mean(get_coordination_numbers(struc.atoms)) for struc in gen_valid_atoms]
            ref_coordination_numbers = [np.mean(get_coordination_numbers(struc.atoms)) for struc in ref_valid_atoms]
            wdist_coordination_numbers = wasserstein_distance(gen_coordination_numbers, ref_coordination_numbers)
            wdist_avg = np.average([wdist_density, wdist_narity, wdist_coordination_numbers])

            if self.dataset_name in ("mp_20", "carbon_24", "perov_5"):
                # Taken from https://github.com/jiaor17/DiffCSP/blob/7121d159826efa2ba9500bf299250d96da37f146/scripts/compute_metrics.py
                COV_Cutoffs = {
                    "mp_20": {"struc": 0.4, "comp": 10.0},
                    "carbon_24": {"struc": 0.2, "comp": 4.0},
                    "perov_5": {"struc": 0.2, "comp": 4}
                }
                struc_cutoff = COV_Cutoffs[self.dataset_name]["struc"]
                comp_cutoff = COV_Cutoffs[self.dataset_name]["comp"]
                cov_precision, cov_recall = get_cov(gen_valid_atoms, ref_valid_atoms, struc_cutoff, comp_cutoff, None)
                cov_avg = np.average([1.0 - cov_precision, 1.0 - cov_recall])
                metrics = {
                    "valid_rate": valid_rate,
                    "wdist_density": wdist_density,
                    "wdist_narity": wdist_narity,
                    "wdist_coordination_numbers": wdist_coordination_numbers,
                    "cov_precision": cov_precision,
                    "cov_recall": cov_recall,
                }
                dng_eval = float(np.average([cov_avg, wdist_avg, 1.0 - valid_rate]))
            else:
                assert self.dataset_name in ["mpts_52", "alex_mp_20"]
                metrics = {
                    "valid_rate": valid_rate,
                    "wdist_density": wdist_density,
                    "wdist_narity": wdist_narity,
                    "wdist_coordination_numbers": wdist_coordination_numbers,
                }
                dng_eval = float(np.average([wdist_avg, 1.0 - valid_rate]))

            self.log("dng_eval", dng_eval, sync_dist=True)
            self.log_dict(metrics, sync_dist=True)
        else:
            assert self._validation_metric == self.ValidationMetric.LOSS

    def predict_step(self, x: OMGData) -> OMGData:
        """
        Performs generation
        """
        x_0 = self.sampler.sample_p_0(x).to(self.device)
        gen, inter = self.si.integrate(x_0, self.model, save_intermediate=True)
        # probably want to turn structure back into some other object that's easier to work with
        filename = (Path(self.generation_xyz_filename) if self.generation_xyz_filename is not None
                    else Path(f"{time.strftime('%Y%m%d-%H%M%S')}.xyz"))
        init_filename = filename.with_stem(filename.stem + "_init")
        xyz_saver(x_0.to("cpu"), init_filename)
        xyz_saver(gen.to("cpu"), filename)
        return gen
