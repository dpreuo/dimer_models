import numpy as np
from koala.lattice import Lattice
from koala import pachner_moves
from functools import partial

def squenergy_function(list_of_sides: np.ndarray):
    distance_from_hex = np.abs(list_of_sides - 6)/2
    return np.sum(distance_from_hex**2)

def lattice_squenergy(lattice):
    """Returns the number of squares in the lattice"""
    sides = np.array([p.n_sides for p in lattice.plaquettes])
    return squenergy_function(sides)

def n_squares(lattice):
    """Returns the number of squares in the lattice"""
    sides = np.array([p.n_sides for p in lattice.plaquettes])
    return np.sum(sides == 4)

def plus_change_squenergy(lattice: Lattice, plaquette: int, central_vertex: int):
    """Returns the change in the number of squares if we perform a plus move on the given plaquette and vertex"""

    plaq = lattice.plaquettes[plaquette]
    assert plaq.n_sides > 4, "You cant contract a square"
    assert central_vertex in plaq.vertices, "The vertex must be on the plaquette given"

    _ = np.where(plaq.vertices == central_vertex)[0][0]
    edges_around = plaq.edges[np.arange(_ - 2, _ + 2) % plaq.n_sides]

    plaqs_to_check = np.array(
        [
            lattice.edges.adjacent_plaquettes[a]
            for a in [edges_around[0], edges_around[3]]
        ]
    ).flatten()

    p, c = np.unique(plaqs_to_check, return_counts=True)

    sides_relevant = np.array([lattice.plaquettes[i].n_sides for i in p])
    new_sides = sides_relevant + 2
    new_sides[np.where(c == 2)] -= 4
    new_sides = np.append(new_sides, 4)  # the new plaquette created has 4 sides


    n_squares_before = squenergy_function(sides_relevant)
    n_squares_after = squenergy_function(new_sides)
    return int(n_squares_after - n_squares_before)


def minus_change_squenergy(lattice: Lattice, edge: int):
    """Returns the change in the number of squares if we perform a minus move on the given edge"""

    side_plaqs = lattice.edges.adjacent_plaquettes[edge]
    vertices = lattice.edges.indices[edge]
    plaqs = np.unique(lattice.vertices.adjacent_plaquettes[vertices].flatten())
    corner_plaqs = np.array(list(set(plaqs) - set(side_plaqs)))

    side_sides = np.array([lattice.plaquettes[x].n_sides for x in side_plaqs])
    corner_sides = np.array([lattice.plaquettes[x].n_sides for x in corner_plaqs])

    sides_after = side_sides - 2
    corner_after = np.sum(corner_sides) - 2

    n_squares_before = squenergy_function(side_sides) + squenergy_function(corner_sides)
    n_squares_after = squenergy_function(sides_after) + squenergy_function(np.array([corner_after]))

    return int(n_squares_after - n_squares_before)


def find_plus_candidate(lattice: Lattice, rng=np.random.default_rng()):
    """Finds a random candidate for a plus move, 
    i.e. a plaquette with more than 4 sides and a vertex on it such that at most one of the adjacent plaquettes is a square"""

    candidate = None
    while candidate is None:
        plaq = rng.choice(len(lattice.plaquettes))
        vertex = int(rng.choice(lattice.plaquettes[plaq].vertices))
        c1 = lattice.plaquettes[plaq].n_sides > 4

        adj_plaqs = lattice.vertices.adjacent_plaquettes[vertex]
        c2 = np.sum([lattice.plaquettes[i].n_sides == 4 for i in adj_plaqs]) <= 1

        if c1 and c2:
            candidate = (plaq, vertex)

    return (plaq, vertex)


def find_minus_candidate(lattice: Lattice, rng=np.random.default_rng(), only_check_squares = True):
    """Finds a random candidate for a minus move, 
    i.e. an edge such that the two plaquettes adjacent to it are not squares and at most one of the other adjacent plaquettes is a square"""

    candidate = None
    while candidate is None:
        plaq = rng.choice(len(lattice.plaquettes))
        if lattice.plaquettes[plaq].n_sides != 4 and only_check_squares:
            continue
        chosen_vertex = rng.choice(lattice.plaquettes[plaq].vertices)
        edges_out = lattice.vertices.adjacent_edges[chosen_vertex]
        chosen_edge = list(set(edges_out) - set(lattice.plaquettes[plaq].edges))[0]

        cond = [
            lattice.plaquettes[i].n_sides == 4
            for i in lattice.edges.adjacent_plaquettes[chosen_edge]
        ]
        if np.sum(cond) == 0:
            candidate = chosen_edge

    return (candidate,)


def boltzmann_probability(delta_energy, beta):
    """Returns the Boltzmann probability of accepting a move with energy change delta_energy at inverse temperature beta"""
    return np.exp(-delta_energy * beta)


def choose_flip(n_each_flip, f_range=10, only_check_squares = True):
    """Given the number of times each flip has been performed, returns a candidate for the next flip, with a bias towards the less performed one."""
    diff = (n_each_flip["plus"] - n_each_flip["minus"]) / f_range + 0.5
    p = np.random.rand()
    if p > diff:
        return (
            find_plus_candidate,
            plus_change_squenergy,
            pachner_moves.bipartite_1_plus,
            "plus",
        )
    else:
        return (
            partial(find_minus_candidate,only_check_squares = only_check_squares),
            minus_change_squenergy,
            pachner_moves.bipartite_1_minus,
            "minus",
        )
