import ctypes

import mpmath
import numpy as np
import numpy.typing as npt
import pfapack.ctypes as cpf
from koala.flux_finder import fluxes_from_ujk, ujk_from_fluxes
from koala.lattice import Lattice


def kasteleyn_matrix(lattice: Lattice, orientation: npt.NDArray, boundary_twist=None):
    """Given an orientation, generate the corresponding Kasteleyn matrix

    Args:
        lattice (Lattice): The lattice
        orientation (npt.NDArray): The orientation
        boundary_twist (np.ndarray, optional): The twist to the x and y boundary. Defaults to None.

    Returns:
        np.ndarray: the Kasteleyn matrix
    """

    dtype = int if boundary_twist is None else complex
    ham = np.zeros((lattice.n_vertices, lattice.n_vertices), dtype=dtype)

    hoppings = orientation.astype(int)

    if boundary_twist is not None:
        phases = np.sum(lattice.edges.crossing * boundary_twist, axis=-1)
        phases = np.exp(1j * phases)
        hoppings = hoppings * phases

    for j, h in enumerate(hoppings):
        ham[lattice.edges.indices[j, 1], lattice.edges.indices[j, 0]] += h
        ham[lattice.edges.indices[j, 0], lattice.edges.indices[j, 1]] -= h.conj()

    return ham


def _fast_pfaffian(K):
    skpf10_d = cpf._init("skpf10_d")
    matrix_f = np.asarray(K, dtype=np.float64, order="F")
    result_array = (ctypes.c_double * 2)(0.0, 0.0)
    uplo_bytes = b"U"
    method_bytes = b"P"
    skpf10_d(K.shape[0], matrix_f, result_array, uplo_bytes, method_bytes)
    return (result_array[0], result_array[1])


def fast_pfaffian_as_mpmath(K):
    m, e = _fast_pfaffian(K)
    return mpmath.mpf(f"{m}e{int(e)}")


def generate_kasteleyn_orientation(lattice: Lattice, defects=None, initial_ujk=None):
    """Make a valid Kasteleyn orientation and check it works

    Args:
        lattice (Lattice): The lattice
        defects(np.ndarray | None): The plaquettes to put defects in

    Returns:
        np.ndarray: A 1d array of kasteleyn orientations
    """
    target_fluxes = np.array([-1] * lattice.n_plaquettes)

    if defects is not None:
        for x in defects:
            target_fluxes[x] *= -1

    ujk = ujk_from_fluxes(lattice, target_fluxes, initial_ujk)
    assert np.all(fluxes_from_ujk(lattice, ujk) == target_fluxes)
    return ujk


################################################
############### Disc algorithms ###############
################################################


################################################
############### Torus algorithms ###############
################################################


def generate_torus_orientations(lattice: Lattice, initial_ujk: np.ndarray = None):
    """Given a lattice, and potentially a starting kasteleyn weighting, generate the four
    kasteleyn weightings corresponding to each PBC sector

    Args:
        lattice (Lattice): A lattice
        initial_ujk (np.ndarray, optional): The initiall choice for a Kasteleyn orientation.
            Defaults to None.

    Returns:
        np.ndarray: A set of four kasteleyn weightings corresponding to each PBC sector
    """

    if initial_ujk is None:
        initial_ujk = generate_kasteleyn_orientation(lattice)

    assert np.all(lattice.boundary_conditions), "This is only necessary on the torus"

    x_boundary_edges = np.where(lattice.edges.crossing[:, 0] != 0)[0]
    y_boundary_edges = np.where(lattice.edges.crossing[:, 1] != 0)[0]
    all_ujks = np.array([initial_ujk] * 4)

    all_ujks[np.ix_([1, 3], x_boundary_edges)] *= -1
    all_ujks[np.ix_([2, 3], y_boundary_edges)] *= -1

    og_flux = fluxes_from_ujk(lattice, initial_ujk)
    assert np.all([np.all(fluxes_from_ujk(lattice, c) == og_flux) for c in all_ujks])

    return all_ujks


def find_omega(pfaffian_vals: np.ndarray):
    """Given a set of four values (generally from Pfaffians) find the orientation of
    omega = perm(1,1,1,-1) that maximises their weighted sum. Useful for determining
    the right sector of the kasteleyn matrix to call K00.

    Args:
        pfaffian_vals (np.ndarray): A set our four numbers to be summed

    Returns:
        np.ndarray: Omega that maximises their sum
    """

    M = (np.ones([4, 4]) - 2 * np.eye(4)).astype(int)

    choices = np.sum((M * pfaffian_vals), axis=1) / 2
    winner = np.argmax(np.abs(choices))
    omega = M[winner]
    return omega


def torus_kasteleyn_number(kasteleyn_matrices):
    """Given a set of four Kasteleyn matrices for each global flux sector, comput the numebr of
    dimerisations of the graph

    Args:
        kasteleyn_matrices (np.ndarray): A kasteleyn matrix for each of the four sectors

    Returns:
        mpmath.ctx_mp_python.mpf : The number of dimerisations
    """
    pfaffians = np.array([fast_pfaffian_as_mpmath(k) for k in kasteleyn_matrices])
    omega = find_omega(pfaffians)
    return mpmath.nint(abs(sum(omega * pfaffians) / 2))


def torus_dimer_probabilities(
    lattice: Lattice,
    kasteleyn_matrices: np.ndarray,
    kasteleyn_inverses: np.ndarray,
    kasteleyn_pfaffians: np.ndarray,
    omega: np.ndarray,
):
    """Given a set of four Kasteleyn matrices, their inverses, and their pfaffians, compute the
    probability of each edge being occupied by a dimer

    Args:
        lattice (Lattice): The lattice object
        kasteleyn_matrices (np.ndarray): The four Kasteleyn matrices for each global flux sector
        kasteleyn_inverses (np.ndarray): The four inverses of the Kasteleyn matrices for each
            global flux sector
        kasteleyn_pfaffians (np.ndarray): The four pfaffians of the Kasteleyn matrices for each
            global flux sector
        omega (np.ndarray): The omega vector that maximises the weighted sum of the pfaffians

    Returns:
        np.ndarray: The probability of each edge being occupied by a dimer
    """
    ncases = (omega * kasteleyn_pfaffians)[:, None, None] * kasteleyn_inverses * kasteleyn_matrices
    inverse_numerator = np.sum(ncases, axis=0)
    inverse_denominator = np.sum(omega * kasteleyn_pfaffians)
    probs_k = (inverse_numerator / inverse_denominator)[*lattice.edges.indices.T]
    return np.abs(probs_k).astype(float)


# TODO - write the disc version of this function
def torus_dimer_correlation(
    lattice: Lattice,
    kasteleyn_matrices: np.ndarray,
    kasteleyn_inverses: np.ndarray,
    kasteleyn_pfaffians: np.ndarray,
    omega: np.ndarray,
    edges_for_corr: np.ndarray,
):
    """Given a set of four Kasteleyn matrices, their inverses, and their pfaffians, compute the
    probability of a set of edges being simultaneously occupied by dimers

    Args:
        lattice (Lattice): The lattice object
        kasteleyn_matrices (np.ndarray): The four Kasteleyn matrices for each global flux sector
        kasteleyn_inverses (np.ndarray): The four inverses of the Kasteleyn matrices for each
            global flux sector
        kasteleyn_pfaffians (np.ndarray): The four pfaffians of the Kasteleyn matrices for each
            global flux sector
        omega (np.ndarray): The omega vector that maximises the weighted sum of the pfaffians
        edges_for_corr (np.ndarray): The edges to compute the correlation for

    Returns:
        float: The probability of the edges being simultaneously occupied by dimers

    """
    verts = lattice.edges.indices[edges_for_corr]
    kasteleyn_products = np.prod(kasteleyn_matrices[:, *verts.T], axis=1)
    slice = np.ix_(verts.flatten(), verts.flatten())
    inverse_chunk = kasteleyn_inverses[:, *slice]
    pfaffians_chunk = np.array([fast_pfaffian_as_mpmath(k) for k in inverse_chunk])
    numerator = np.sum(omega * kasteleyn_pfaffians * pfaffians_chunk * kasteleyn_products)
    inverse_denominator = np.sum(omega * kasteleyn_pfaffians)
    probability = np.abs(numerator / inverse_denominator)
    return probability


def torus_monomer_count(
    kasteleyn_tilde_matrices: np.ndarray,
    inverses_tilde: np.ndarray,
    pfaffians_tilde: np.ndarray,
    chosen_vertices: np.ndarray,
):
    """Given a set of four Kasteleyn matrices, their inverses, and their pfaffians, compute the
    number of dimerisations with monomers at the chosen vertices

    Args:
        kasteleyn_tilde_matrices (np.ndarray): The four Kasteleyn matrices for each
            global flux sector
        inverses_tilde (np.ndarray): The four inverses of the Kasteleyn matrices for each
            global flux sector
        pfaffians_tilde (np.ndarray): The four pfaffians of the Kasteleyn matrices for each
            global flux sector
        chosen_vertices (np.ndarray): The vertices to compute the monomer count for

    Returns:
        mpmath.ctx_mp_python.mpf : The number of dimerisations with monomers at the chosen vertices
    """

    inverses_tilde_restricted = inverses_tilde[:, *np.ix_(chosen_vertices, chosen_vertices)]
    inverses_tilde_pfaffs = np.array(
        [fast_pfaffian_as_mpmath(k) for k in inverses_tilde_restricted]
    )
    omega_tilde = find_omega(inverses_tilde_pfaffs * pfaffians_tilde)
    monomer_numerator = omega_tilde * inverses_tilde_pfaffs * pfaffians_tilde
    n_monomers_pfaff = mpmath.nint(mpmath.fabs(sum(monomer_numerator) / 2))

    # this doesnt work if one pfaffian vanishes and we have to find the full reduced matrix
    if np.any(pfaffians_tilde < 1e-10) and n_monomers_pfaff != 0:
        n = kasteleyn_tilde_matrices.shape[1]
        mask = np.delete(np.arange(n), chosen_vertices)
        kasteleyn_reduced = kasteleyn_tilde_matrices[:, *np.ix_(mask, mask)]
        n_monomers_pfaff = torus_kasteleyn_number(kasteleyn_reduced)

    return n_monomers_pfaff


# TODO - the sign of this doesnt always agree with brute force -- is this a problem?
def torus_vison_correlation(
    pfaffians_tilde: np.ndarray,
    omega: np.ndarray,
):
    """Given a set of four Kasteleyn matrices, their inverses, and their pfaffians, compute the
    weighted sum over dimerisations with visons at the chosen plaquettes

    Args:
        pfaffians_tilde (np.ndarray): The four pfaffians of the Kasteleyn matrices for each
            global flux sector
        omega (np.ndarray): The omega vector that maximises the weighted sum of the pfaffians

    Returns:
        mpmath.ctx_mp_python.mpf : The number of dimerisations with visons
    """

    vison_numerator = omega * pfaffians_tilde
    n_vison_pfaff = mpmath.nint(sum(vison_numerator) / 2)
    return abs(n_vison_pfaff)


##################################################
##############  frontend functions  ##############
##################################################


def find_kasteleyn_number(lattice: Lattice):
    """Given a lattice, find the total number of dimerisations using Kasteleyn's method.

    Args:
        lattice (Lattice): The lattice object, must have periodic or open boundaries.

    Raises:
        ValueError: Kasteleyn only works in periodic and open boundaries


    Returns:
        mpmath.ctx_mp_python.mpf : The number of dimerisations
    """

    # check boundaries
    if np.all(lattice.boundary_conditions):
        orientations = generate_torus_orientations(lattice)
        kasteleyn_matrices = np.array([kasteleyn_matrix(lattice, o) for o in orientations])
        return torus_kasteleyn_number(kasteleyn_matrices)
    elif np.all(~lattice.boundary_conditions):
        orientation = generate_kasteleyn_orientation(lattice)
        k_matrix = kasteleyn_matrix(lattice, orientation)
        return fast_pfaffian_as_mpmath(k_matrix)
    else:
        raise ValueError("Only works if the latttice is in full PBC or OBC")
