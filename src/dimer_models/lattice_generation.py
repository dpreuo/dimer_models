
from koala.pointsets import uniform
from koala import graph_utils as gu
from koala.voronization import generate_lattice
import numpy as np
from koala.lattice import Lattice, INVALID
from koala.graph_utils import remove_vertices
from copy import copy

def bipartite_squarefull(n_sites, ensure_true_bipartite = True):

    check = True
    while check:
        lattice = generate_lattice(uniform(n_sites//4))
        dimerisation = gu.dimerise(lattice)
        lattice = gu.dimer_collapse(lattice, dimerisation)
        out = gu.vertices_to_polygon(lattice)

        if not ensure_true_bipartite:
            break
        
        x_loop_len = len(gu.find_periodic_loop(out, 'x'))
        y_loop_len = len(gu.find_periodic_loop(out, 'y'))
        
        if x_loop_len%2 + y_loop_len%2 == 0:
            check = False

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

        crossing_to_add = (
            crossing_edge + sign_in * starting_cross + sign_out * ending_cross
        )

        edges_to_keep = np.concatenate([edges_to_keep, edge_to_add[None, :]])
        crossing_to_keep = np.concatenate([crossing_to_keep, crossing_to_add[None, :]])

    l_out = Lattice(positions, edges_to_keep, crossing_to_keep)
    l_out = remove_vertices(l_out, newly_hinge_vertices)

    return l_out

def reduce_bipartite(lattice:Lattice, n_steps = None):

    if n_steps==None: 
        r = 1000
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