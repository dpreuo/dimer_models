from koala.lattice import Lattice
import koala.graph_utils as gu
import numpy as np

def bipartition(lattice: Lattice) -> np.ndarray:
    """Construct a bipartition of a given lattice, raises error if the lattice is not bipartite

    Args:
        lattice (Lattice): The bipartite lattice

    Raises:
        Exception: Lattice is not bipartite

    Returns:
        np.ndarray: _description_
    """

    bp = gu.distance_matrix(lattice)[0] % 2

    edge_colours = np.sum(bp[lattice.edges.indices], axis=1)

    if not np.all(edge_colours == 1):
        raise Exception("Lattice is not bipartite")

    return bp


def dimer_height(
    lattice: Lattice,
    dimerisation: np.ndarray,
    bipart: np.ndarray = None,
    avoid_boundaries: bool = False,
) -> np.ndarray:
    """Given a lattice and a dimerisation, generate the heights corresponding to said dimerisation.

    Args:
        lattice (Lattice): The lattice
        dimerisation (np.ndarray): A dimerisation of the lattice
        bipart (np.ndarray, optional): You can provide a bipartition of the lattice.
             Defaults to None.
        avoid_boundaries (bool, optional): The tree on which the heights are calculated
            can avoid the periodic boundaries if this is set to false. Defaults to False.

    Raises:
        Exception: A bug has appeared when trying to assign default bond directions
        Exception: Lattice is not regular

    Returns:
        np.ndarray: The height of each plaquette
    """

    if bipart is None:
        bipart = bipartition(lattice)

    directions = 1 - 2 * bipart[lattice.edges.indices][:, 0]
    dual = gu.make_dual(lattice, reg_steps=0)

    # check it's regular
    coordination = lattice.vertices.coordination_numbers
    if not np.all(coordination == coordination[0]):
        raise Exception("Lattice is not regular")

    # check that each dual plaquette has the right swirling
    for i, p in enumerate(dual.plaquettes):
        pols = directions[p.edges] * p.directions
        if not np.all(pols == pols[0]):
            raise Exception(
                "A bug has appeared when trying to assign default bond directions"
            )
        
    coordination = coordination[0]

    # make spanning tree
    tree = gu.edge_spanning_tree(dual, cross_boundaries=avoid_boundaries)
    tree_edges = dual.edges.indices[tree]
    tree_dirs = directions[tree]
    tree_dimers = dimerisation[tree]

    dimer_cost = coordination - 1
    non_dimer_cost = -1

    tree_height_diffs = tree_dirs * (
        dimer_cost * (tree_dimers) + non_dimer_cost * (1 - tree_dimers)
    )

    heights = np.full(lattice.n_plaquettes, None)
    heights[0] = 0

    # compute all heights
    while np.any(heights == None):
        heights_assigned = heights != None

        assign_index_in_tree = np.where(
            np.sum(heights_assigned[tree_edges], axis=1) % 2
        )
        assignable_edges = tree_edges[assign_index_in_tree]
        ass_tree_height_diffs = tree_height_diffs[assign_index_in_tree]

        edge_signs = -1 + 2 * np.argwhere(heights_assigned[assignable_edges])[:, 1]

        vertices_to_assign = assignable_edges[
            np.where(1 - heights_assigned[assignable_edges])
        ]
        where_we_came_from = assignable_edges[
            np.where(heights_assigned[assignable_edges])
        ]

        heights[vertices_to_assign] = (
            heights[where_we_came_from] + ass_tree_height_diffs * edge_signs
        )

    heights = heights - np.min(heights)

    return heights