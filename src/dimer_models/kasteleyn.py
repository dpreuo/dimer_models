from koala.lattice import Lattice
import numpy as np
import numpy.typing as npt
from koala.flux_finder import fluxes_from_ujk, ujk_from_fluxes
from pfapack.ctypes import pfaffian
import pfapack.ctypes as cpf
import ctypes
from scipy import linalg as la


def kasteleyn_matrix(lattice: Lattice, ujk: npt.NDArray):

    ham = np.zeros((lattice.n_vertices, lattice.n_vertices))
    hoppings = ujk

    ham[lattice.edges.indices[:, 1], lattice.edges.indices[:, 0]] = hoppings
    ham[lattice.edges.indices[:, 0], lattice.edges.indices[:, 1]] = -1 * hoppings

    return ham

def _pull_edge_entries(lattice, matrix):

    out = np.zeros(lattice.n_edges)
    for j, edge in enumerate(lattice.edges.indices):
        e = np.sort(edge)
        out[j] = matrix[tuple(e)]
    return out


def _fast_pfaffian(K):
    skpf10_d = cpf._init("skpf10_d")
    matrix_f = np.asarray(K, dtype=np.float64, order="F")
    result_array = (ctypes.c_double * 2)(0.0, 0.0)
    uplo_bytes = "U".encode()
    method_bytes = "P".encode()
    skpf10_d(K.shape[0], matrix_f, result_array, uplo_bytes, method_bytes)
    return (result_array[0], result_array[1])


def find_kasteleyn_number(lattice: Lattice, s_log=False):
    """Given a lattice, find the total number of dimerisations using Kasteleyn's method.

    Args:
        lattice (Lattice): The lattice object, must have operiodic or open boundaries.
        s_log (bool, optional): If True, returns the result as a tuple where 
            out[0]* 10 ** out[1] is the number of dimerisations. Defaults to False.

    Raises:
        Exception: Kasteleyn only works in periodic and open boundaries


    Returns:
        int : _description_
    """

    # open boundaries
    if np.all(lattice.boundary_conditions == False):
        ujk = ujk_from_fluxes(lattice, np.array([-1] * lattice.n_plaquettes))
        assert np.all(fluxes_from_ujk(lattice, ujk) == -1)
        K = kasteleyn_matrix(lattice, ujk)

        if s_log:
            out = _fast_pfaffian(K)
            return (abs(out[0]), out[1])
        else:
            return abs(round(pfaffian(K)))

    # periodic boundaries
    elif np.all(lattice.boundary_conditions == True):

        # make ujk sets
        ujk = ujk_from_fluxes(lattice, np.array([-1] * lattice.n_plaquettes))
        x_boundary_edges = np.where(lattice.edges.crossing[:, 0] != 0)
        y_boundary_edges = np.where(lattice.edges.crossing[:, 1] != 0)
        all_ujks = np.array([[ujk] * 2] * 2)
        all_ujks[1, :, x_boundary_edges] *= -1
        all_ujks[:, 1, y_boundary_edges] *= -1
        ujk_list = all_ujks.reshape([-1, all_ujks.shape[-1]])

        n_vals = []
        for u in ujk_list:

            # solve the pfaffians
            assert np.all(fluxes_from_ujk(lattice, u) == -1)
            K = kasteleyn_matrix(lattice, u)
            if s_log:
                n_vals.append(_fast_pfaffian(K))
            else:
                n_vals.append(round(pfaffian(K)))

        # find the right combination to return
        if s_log:
            mags = np.array([x[0] for x in n_vals])
            pows = np.array([x[1] for x in n_vals])

            max_power = max(pows)
            defecit = pows - max_power
            mags = mags * (10 ** (defecit))

            m = np.ones([4, 4]) - 2 * np.eye(4)
            options = np.abs(np.sum(m * np.array(mags), axis=1)) / 2

            return np.max(options), max_power

        else:
            m = np.ones([4, 4]) - 2 * np.eye(4)
            options = np.abs(np.sum(m * np.array(n_vals), axis=1)) // 2

            return int(np.max(options))

    else:
        raise Exception("Kasteleyn only works in periodic and open boundaries")


def find_local_dimer_probability(lattice: Lattice):
    """Given a lattice, find the probability of each edge hosting a dimer

    Args:
        lattice (Lattice): The lattice object, must have operiodic or open boundaries.

    Raises:
        Exception: Kasteleyn only works in periodic and open boundaries


    Returns:
        np.ndarray: The probability of each edge hosting a dimer.
    """

    # open boundaries
    if np.all(lattice.boundary_conditions == False):
        ujk = ujk_from_fluxes(lattice, np.array([-1] * lattice.n_plaquettes))
        K = kasteleyn_matrix(lattice, ujk)
        kinv = la.inv(K)
        return np.abs(_pull_edge_entries(lattice, kinv))

    elif np.all(lattice.boundary_conditions == True):
        # make ujk sets
        ujk = ujk_from_fluxes(lattice, np.array([-1] * lattice.n_plaquettes))
        x_boundary_edges = np.where(lattice.edges.crossing[:, 0] != 0)
        y_boundary_edges = np.where(lattice.edges.crossing[:, 1] != 0)
        all_ujks = np.array([[ujk] * 2] * 2)
        all_ujks[1, :, x_boundary_edges] *= -1
        all_ujks[:, 1, y_boundary_edges] *= -1
        ujk_list = all_ujks.reshape([-1, all_ujks.shape[-1]])

        # find pfaffians and inverses
        n_mags = np.zeros(4)
        n_pows = np.zeros(4)
        kinv_vals = np.zeros([4, lattice.n_vertices, lattice.n_vertices])
        for i, u in enumerate(ujk_list):
            assert np.all(fluxes_from_ujk(lattice, u) == -1)
            K = kasteleyn_matrix(lattice, u)
            n_mags[i], n_pows[i] = _fast_pfaffian(K)
            
            if n_mags[i] == 0:
                continue
            
            kinv_vals[i] = (
                la.inv(K) * K
            )  # I have no idea why multiplying with K works but it does...

        # find the total number of dimerisations
        max_power = max(n_pows)
        defecit = n_pows - max_power
        n_mags = n_mags = n_mags * (10 ** (defecit))
        m = np.ones([4, 4]) - 2 * np.eye(4)
        options = np.abs(np.sum(m * np.array(n_mags), axis=1))
        w = m[np.argmax(options)]
        total_mag = np.max(options)

        # multiply probability with weight and normalise
        inv_total = np.einsum("i, ijk, i -> jk", w, kinv_vals, n_mags)
        result_mag = np.abs(_pull_edge_entries(lattice, inv_total)) / total_mag

        return result_mag
    else:
        raise Exception("Kasteleyn only works in periodic and open boundaries")