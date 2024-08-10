import quimb as qu
import quimb.tensor as qtn
import random


random.seed(40)
qu.gen.rand.seed_rand(40)


def random_disjoint_matching(N, prev_pairs):
    prev_pairs = prev_pairs.copy()
    random.shuffle(prev_pairs)
    for i in range(len(prev_pairs)):
        prev_pairs[i] = list(prev_pairs[i])
        random.shuffle(prev_pairs[i])
    
    out = []
    if N % 2 == 0:
        head, tail = prev_pairs.pop()
    else:
        # Missing index
        head = (N * (N - 1) // 2) - sum(sum(pair) for pair in prev_pairs)
        
    for i, j in prev_pairs:
        out.append((head, i))
        head = j

    if N % 2 == 0:
        out.append((head, tail))

    return out
    

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
    pairs = [((s + i) % N, (s + i + 1) % N) for i in range(0, N - 1, 2)]
    return pairs


def single_qubit_layer(circ, gate_round=None):
    """Apply a parametrizable layer of single qubit ``U3`` gates."""
    for i in range(circ.N):
        # initialize with random parameters
        params = qu.randn(3, dist="uniform")
        circ.apply_gate("U3", *params, i, gate_round=gate_round, parametrize=True)


def two_qubit_layer(circ, pairs, gate_round=None):
    """Apply aparametrizable layer of two qubit ``RZZ`` gates."""
    for i, j in pairs:
        # initialize with random parameters
        params = qu.randn(1, dist="uniform")
        circ.apply_gate("RZZ", *params, i, j, gate_round=gate_round, parametrize=True)


def ansatz_circuit(n, depth, two_qubit_mode="random_matching", **kwargs):
    """Construct a circuit of single qubit and entangling layers."""
    assert n > 1
    circ = qtn.Circuit(n, **kwargs)

    for gate_round in range(depth):
        # single qubit gate layer
        single_qubit_layer(circ, gate_round=gate_round)

        if two_qubit_mode == "random_matching":
            pairs = random_matching(circ.N)
        elif two_qubit_mode == "brickwork":
            pairs = brickwork(circ.N, s=gate_round)
        elif two_qubit_mode == "random_disjoint":
            if gate_round == 0:
                pairs = random_matching(circ.N)
            else:
                pairs = random_disjoint_matching(circ.N, pairs)
            
        # apply RZZ layers
        two_qubit_layer(circ, pairs, gate_round=gate_round)

    # add a final single qubit layer
    single_qubit_layer(circ, gate_round=depth + 1)

    return circ
