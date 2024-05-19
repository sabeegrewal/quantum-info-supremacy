import quimb as qu
import quimb.tensor as qtn
import random


random.seed(40)
qu.gen.rand.seed_rand(40)


def random_matching(N):
    """Generate a random maximal set of disjoint pairs from the set {0, ..., N-1}"""
    S = set(range(N))
    out = []
    while len(S) >= 2:
        i = random.choice(list(S))
        S.remove(i)
        j = random.choice(list(S))
        S.remove(j)
        out.append((i, j))
    return out


def brickwork(N, s):
    pairs = [((s + i) % N, (s + i + 1) % N) for i in range(0, N, 2)]
    if N % 2 != 0:
        pairs.pop()
    return pairs


def single_qubit_layer(circ, gate_round=None):
    """Apply a parametrizable layer of single qubit ``U3`` gates."""
    for i in range(circ.N):
        # initialize with random parameters
        params = qu.randn(3, dist="uniform")
        circ.apply_gate("U3", *params, i, gate_round=gate_round, parametrize=True)


def two_qubit_layer(circ, two_qubit_mode="random_matching", gate_round=None):
    """Apply aparametrizable layer of two qubit ``RZZ`` gates."""
    if two_qubit_mode is "random_matching":
        pairs = random_matching(circ.N)
    if two_qubit_mode is "brickwork":
        assert gate_round is not None
        pairs = brickwork(circ.N, s=gate_round)

    for i, j in pairs:
        # initialize with random parameters
        params = qu.randn(1, dist="uniform")
        circ.apply_gate("RZZ", *params, i, j, gate_round=gate_round, parametrize=True)


def ansatz_circuit(n, depth, two_qubit_mode="random_matching", **kwargs):
    """Construct a circuit of single qubit and entangling layers."""
    assert n > 1
    circ = qtn.Circuit(n, **kwargs)

    for r in range(depth):
        # single qubit gate layer
        single_qubit_layer(circ, gate_round=r)

        # alternate between forward and backward CZ layers
        two_qubit_layer(circ, two_qubit_mode, gate_round=r)

    # add a final single qubit layer
    single_qubit_layer(circ, gate_round=r + 1)

    return circ
