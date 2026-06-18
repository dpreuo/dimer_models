from collections.abc import Iterable
from copy import copy

import numpy as np
from koala import graph_utils as gu
from koala.graph_utils import remove_vertices
from koala.lattice import INVALID, Lattice
from koala.pointsets import uniform
from koala.voronization import generate_lattice


def expand_edges(lattice: Lattice, edge_indices: np.ndarray) -> Lattice:
    """Expands a set of n edges into plaquettes, removing the two vertices that touch the given edge

    Args:
        lattice (Lattice): The lattice
        edge_indices (np.ndarray): array of all edges to be removed

    Raises:
        Exception: List of edges to remove is intersecting

    Returns:
        Lattice: The new lattice with edges removed
    """

    if not isinstance(edge_indices, Iterable):
        edge_indices = np.array([edge_indices])

    if len(edge_indices) == 0:
        return lattice

    positions = lattice.vertices.positions
    edges = lattice.edges.indices
    crossing = lattice.edges.crossing

    adjacent_edges = lattice.edges.adjacent_edges

    vertices_for_removal = []
    edges_to_add = []
    crossing_to_add = []
    for ed in edge_indices:
        # check the vertices have not already been marked for removal
        if len(np.intersect1d(vertices_for_removal, lattice.edges.indices[ed])):
            raise Exception("List of edges to remove is intersecting")

        # add to the list
        vertices_for_removal.append(lattice.edges.indices[ed])

        # now we add each new edge
        adjacent_plaqs = lattice.edges.adjacent_plaquettes[ed]
        for p in adjacent_plaqs:
            plaq = lattice.plaquettes[p]
            chosen_edge = np.where(plaq.edges == ed)[0][0]

            edges_in_triad = np.roll(plaq.edges, -chosen_edge + 1)[:3]
            directions = np.roll(plaq.directions, -chosen_edge + 1)[:3]

            vertices_in_triad = lattice.edges.indices[edges_in_triad]
            vertices_in_triad[np.where(directions == -1)] = vertices_in_triad[
                np.where(directions == -1)
            ][:, ::-1]

            new_edge = np.array([vertices_in_triad[0, 0], vertices_in_triad[-1, -1]])
            new_crossing = np.sum(
                directions[:, None] * lattice.edges.crossing[edges_in_triad], axis=0
            )

            if len(np.intersect1d(vertices_for_removal, new_edge)):
                raise Exception("List of edges to remove is intersecting")

            edges_to_add.append(new_edge)
            crossing_to_add.append(new_crossing)

    # make the new lattice with all added edges
    positions = lattice.vertices.positions
    final_edges = np.concatenate([lattice.edges.indices, edges_to_add])
    final_crossing = np.concatenate([lattice.edges.crossing, crossing_to_add])

    # remove the bad vertices
    vertices_for_removal = np.concatenate(vertices_for_removal)
    x = gu._remove_vertices_backend(positions, final_edges, final_crossing, vertices_for_removal)

    return Lattice(*x)


def bipartite_squarefull(n_sites, ensure_true_bipartite=True, return_pre_expanded=False):

    check = True
    while check:
        lattice = generate_lattice(uniform(n_sites // 4))
        dimerisation = gu.dimerise(lattice)
        lattice = gu.dimer_collapse(lattice, dimerisation)
        out = gu.vertices_to_polygon(lattice)

        if not ensure_true_bipartite:
            break

        x_loop_len = len(gu.find_periodic_loop(out, "x"))
        y_loop_len = len(gu.find_periodic_loop(out, "y"))

        if x_loop_len % 2 + y_loop_len % 2 == 0:
            check = False

    if return_pre_expanded:
        return out, lattice
    return out


def find_expandable_plaquettes(lattice: Lattice):

    expandable_plaquettes = []
    for j, plaq in enumerate(lattice.plaquettes):
        if np.any(plaq.adjacent_plaquettes == INVALID):
            continue
        neighbours = lattice.plaquettes[plaq.adjacent_plaquettes]
        neighbouring_sides = np.array([n.n_sides for n in neighbours])

        c1 = np.all(neighbouring_sides[::2] == 4) and np.all(neighbouring_sides[1::2] != 4)
        c2 = np.all(neighbouring_sides[1::2] == 4) and np.all(neighbouring_sides[::2] != 4)
        if c1 or c2:
            expandable_plaquettes.append(j)

    return np.array(expandable_plaquettes)


def expand_plaquette_square_boundary(lattice, plaquette_id):

    chosen_plaq = lattice.plaquettes[plaquette_id]
    neighbours = lattice.plaquettes[chosen_plaq.adjacent_plaquettes]
    sides = np.array([n.n_sides for n in neighbours])
    first_4 = np.argwhere(sides == 4)[0, 0]
    edges_for_removal = chosen_plaq.edges[first_4::2]
    edges_for_expansion = chosen_plaq.edges[(1 - first_4) :: 2]
    positions = lattice.vertices.positions.copy()

    # remove the edges that should be deleted
    edges_to_keep = np.delete(lattice.edges.indices, edges_for_removal, axis=0)
    crossing_to_keep = np.delete(lattice.edges.crossing, edges_for_removal, axis=0)
    newly_hinge_vertices = chosen_plaq.vertices

    # now add the new edges
    for e in edges_for_expansion:
        edge_out = lattice.edges.adjacent_edges[e]
        edge_out = edge_out[~np.isin(edge_out, edges_for_removal)]

        indices_edge = lattice.edges.indices[e]
        indices_out = lattice.edges.indices[edge_out]

        starting_arg = np.where(indices_out == indices_edge[0])
        ending_arg = np.where(indices_out == indices_edge[1])
        edge_to_add = np.array(
            [
                indices_out[starting_arg[0], 1 - starting_arg[1]],
                indices_out[ending_arg[0], 1 - ending_arg[1]],
            ]
        ).flatten()

        sign_in = -(1 - 2 * starting_arg[1][0])
        sign_out = 1 - 2 * ending_arg[1][0]
        crossing_edge = lattice.edges.crossing[e]
        crossing_out = lattice.edges.crossing[edge_out]
        starting_cross = crossing_out[starting_arg[0]][0]
        ending_cross = crossing_out[ending_arg[0]][0]

        crossing_to_add = crossing_edge + sign_in * starting_cross + sign_out * ending_cross

        edges_to_keep = np.concatenate([edges_to_keep, edge_to_add[None, :]])
        crossing_to_keep = np.concatenate([crossing_to_keep, crossing_to_add[None, :]])

    l_out = Lattice(positions, edges_to_keep, crossing_to_keep)
    l_out = remove_vertices(l_out, newly_hinge_vertices)

    return l_out


def reduce_bipartite(lattice: Lattice, n_steps=None):

    if n_steps is None:
        r = 1000000
    else:
        r = n_steps
    reduced_lattice = copy(lattice)

    for n in range(r):
        options = find_expandable_plaquettes(reduced_lattice)

        if len(options) == 0:
            break

        side_lengths = [p.n_sides for p in reduced_lattice.plaquettes[options]]

        n_sides = np.array([p.n_sides for p in reduced_lattice.plaquettes[options]])
        first_choice = np.random.choice(options)
        x = reduced_lattice.plaquettes[first_choice].adjacent_plaquettes
        reduced_lattice = expand_plaquette_square_boundary(reduced_lattice, first_choice)

    return reduced_lattice


def expand_edges_to_squares(lattice: Lattice, chosen_edges: np.ndarray) -> Lattice:
    """Expand a select number of edges into squares

    Args:
        lattice (Lattice): The starting lattice
        chosen_edges (np.ndarray): The edges to expand

    Returns:
        Lattice: The final lattice
    """

    ind_touched = lattice.edges.indices[chosen_edges].flatten()
    assert len(set(ind_touched)) == len(ind_touched), "Edges to expand touch the same vertex"

    if len(chosen_edges) == 0:
        return lattice

    # make vertices into triangles
    l2 = gu.vertices_to_polygon(lattice, lattice.edges.indices[chosen_edges].flatten())

    # next add the right edges to have the squares present
    edges_to_add = copy(l2.edges.indices)
    crossing_to_add = copy(l2.edges.crossing)
    vertices_to_remove = []

    for chosen_edge in chosen_edges:
        adj_plaqs = l2.edges.adjacent_plaquettes[chosen_edge]
        vertices_to_remove.append(l2.edges.indices[chosen_edge])

        for j, p_index in enumerate(adj_plaqs):
            plaquette = l2.plaquettes[p_index]

            e_in_loop = np.where(plaquette.edges == chosen_edge)[0]
            if len(e_in_loop) > 1:
                e_in_loop = e_in_loop[j]

            i_in_loop = np.array([e_in_loop - 1, e_in_loop, e_in_loop + 1]).flatten() % len(
                plaquette.edges
            )

            signs_in_run = plaquette.directions[i_in_loop]
            edges_in_run = plaquette.edges[i_in_loop]

            vertices_in_run = l2.edges.indices[edges_in_run]
            crossing_in_run = l2.edges.crossing[edges_in_run] * signs_in_run[:, None]

            cross = np.sum(crossing_in_run, axis=0)
            i = (1 - signs_in_run) // 2
            edge = np.array([vertices_in_run[0, i[0]], vertices_in_run[-1, 1 - i[0]]])

            edges_to_add = np.vstack([edges_to_add, edge])
            crossing_to_add = np.vstack([crossing_to_add, cross])

            # crossing_to_add.append(cross)

    l3 = Lattice(l2.vertices.positions, edges_to_add, crossing_to_add)
    l4 = gu.remove_vertices(l3, np.array(vertices_to_remove).flatten())

    return l4
