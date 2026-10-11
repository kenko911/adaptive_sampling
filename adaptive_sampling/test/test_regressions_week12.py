import numpy as np
import pytest
import torch

from adaptive_sampling.colvars import CV
from adaptive_sampling.interface.interfaceASE import AseMD
from adaptive_sampling.interface.interfaceMD_2D import MD as MD2D
from adaptive_sampling.interface.sampling_data import SamplingData
from adaptive_sampling.processing_tools.mbar import pmf_from_weights
from adaptive_sampling.processing_tools.thermodynamic_integration import czar, integrate
from adaptive_sampling.sampling_tools.abf import ABF
from adaptive_sampling.sampling_tools.metadynamics import WTM
from adaptive_sampling.sampling_tools.opes import OPES
from adaptive_sampling.units import (
    DEGREES_per_RADIAN,
    R_in_SI,
    atomic_to_kJmol,
    kB_in_atomic,
)


class MolecularMD:
    def __init__(self, coords):
        self.coords = np.asarray(coords, dtype=float)
        self.mass = np.ones(len(self.coords) // 3)
        self.forces = np.zeros_like(self.coords)
        self.step = 0

    def get_sampling_data(self):
        return SamplingData(
            self.mass,
            self.coords,
            self.forces,
            0.0,
            0.0,
            len(self.mass),
            self.step,
            0.0,
        )


def test_cumulative_trapezoid_uses_bin_centers():
    np.testing.assert_allclose(
        integrate(np.array([1.0, 2.0, 3.0]), 0.5, normalize=False), [0.0, 0.75, 2.0]
    )


def test_czar_uses_centered_grid():
    grid = np.array([0.5, 1.5, 2.5])
    np.testing.assert_allclose(czar(grid, grid.copy(), grid.copy(), sigma=1.0), 0.0)


def test_mbar_pmf_works_with_numpy_2():
    pmf, rho = pmf_from_weights(np.array([0.5, 1.5]), np.array([0.5]), np.array([1.0]))
    assert pmf.mask[1]
    np.testing.assert_allclose(rho, [1.0, 0.0])


def test_gas_constant_matches_sampling_thermal_energy():
    assert R_in_SI / 1000.0 == pytest.approx(kB_in_atomic * atomic_to_kJmol, rel=1e-12)


def test_torsion_type_and_periodicity_are_consistent():
    md = MolecularMD([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
    bias = ABF(
        md,
        [["torsion", [0, 1, 2, 3], -180, 180, 10]],
        periodicity=[[-180, 180]],
        verbose=False,
    )
    assert bias.cv_type == ["torsion"]
    np.testing.assert_allclose(bias.periodicity[0], [-np.pi, np.pi])
    assert bias.dx[0] == pytest.approx(10 / DEGREES_per_RADIAN)


def test_force_constant_conversion_is_per_coordinate():
    md = MolecularMD([0, 1, 0, 0, 0, 0, 1, 0, 0])
    bias = ABF(md, [["angle", [0, 1, 2], 0, 180, 10]], f_conf=1.0, verbose=False)
    assert bias.f_conf[0] == pytest.approx(DEGREES_per_RADIAN**2 / atomic_to_kJmol)


def test_two_dimensional_temperature_uses_two_degrees_of_freedom():
    md = MD2D()
    md.momenta[:] = [1.0, 1.0]
    md.calc_etvp()
    assert md.temp == pytest.approx(md.ekin / kB_in_atomic)


def test_ase_temperature_uses_molecular_degrees_of_freedom(tmp_path):
    class Atoms:
        def get_positions(self):
            return np.ones((22, 3))

        def get_masses(self):
            return np.ones(22)

    md = AseMD(Atoms(), scratch_dir=str(tmp_path))
    md.momenta[:] = 1.0
    md.calc_etvp()
    assert md.ndegrees == 60
    assert md.temp == pytest.approx(2 * md.ekin / (60 * kB_in_atomic))


@pytest.mark.parametrize("method", [WTM, OPES])
def test_histogram_accumulates_without_output_file(method):
    md = MolecularMD([0, 0, 0, 1, 0, 0])
    cv_def = [["distance", [0, 1], 0, 2, 0.1]]
    if method is WTM:
        bias = WTM(
            md, cv_def, hill_height=1, hill_std=0.1, hill_drop_freq=100, verbose=False
        )
    else:
        bias = OPES(md, cv_def, kernel_std=0.1, update_freq=100, verbose=False)
    bias.step_bias(traj_file=None, out_file=None, restart_file=None)
    assert bias.histogram.sum() == 1


def test_wtm_pmf_is_kj_per_mol():
    md = MolecularMD([0, 0, 0, 1, 0, 0])
    bias = WTM(
        md,
        [["distance", [0, 1], 0, 2, 0.1]],
        hill_height=1,
        hill_std=0.1,
        verbose=False,
    )
    bias.metapot[0, :2] = [0.0, 1.0 / atomic_to_kJmol]
    assert bias.get_pmf()[0, 0] == pytest.approx(1.0)


def test_opes_explore_pmf_has_fixed_point_scale_and_kj_units():
    md = MolecularMD([0, 0, 0, 1, 0, 0])
    bias = OPES(
        md,
        [["distance", [0, 1], 0, 2, 0.1]],
        kernel_std=0.1,
        explore=True,
        bias_factor=10,
        verbose=False,
    )
    bias.bias_potential[0, :2] = [0.0, -1.0 / atomic_to_kJmol]
    assert bias.get_pmf()[0, 1] == pytest.approx(10 / 9)


def test_angle_abf_divergence_matches_cartesian_field():
    coords = np.array([[1.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 1.2, 0.1]])
    md = MolecularMD(coords.ravel())
    bias = ABF(md, [["angle", [0, 1, 2], 0, 180, 5]], verbose=False)
    bias.get_cv()

    x = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
    a, b = x[0] - x[1], x[2] - x[1]
    xi = torch.acos(torch.dot(a, b) / (a.norm() * b.norm()))
    (g,) = torch.autograd.grad(xi, x, create_graph=True)
    v = g / g.square().sum()
    expected = sum(
        torch.autograd.grad(v.flatten()[i], x, retain_graph=True)[0].flatten()[i]
        for i in range(9)
    )
    assert bias.cv_divergence[0] == pytest.approx(float(expected), rel=1e-5)


def test_torsion_abf_divergence_matches_finite_difference():
    coords = np.array(
        [[0.1, 1.1, 0.2], [0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.5, 0.8, 0.6]]
    )
    md = MolecularMD(coords.ravel())
    bias = ABF(md, [["torsion", [0, 1, 2, 3], -180, 180, 5]], verbose=False)
    bias.get_cv()
    cv = CV(md, requires_grad=True)

    def inverse_gradient(x):
        md.coords[:] = x
        _, gradient = cv.get_cv("torsion", [0, 1, 2, 3])
        return gradient / np.dot(gradient, gradient)

    expected = 0.0
    step = 1e-3
    for i in range(len(md.coords)):
        plus = coords.ravel().copy()
        minus = coords.ravel().copy()
        plus[i] += step
        minus[i] -= step
        expected += (inverse_gradient(plus)[i] - inverse_gradient(minus)[i]) / (
            2 * step
        )
    assert bias.cv_divergence[0] == pytest.approx(expected, abs=1e-3)
