import random
import functools
import numpy as np


def eta(n, d):
    """Computes the function eta defined Eq. (80) of
    Bravyi-Gosset: https://arxiv.org/pdf/1601.07601

    Parameters
    ----------
    n : int
        Number of qubits in the stabilizer state.
    d : int
        Codimension of the affine subspace.

    Returns
    -------
    float
        A real number proportional to the number of n-qubit
        stabilizer states whose support is an affine subspace
        of dimension n-d.
    """

    result = 2 ** (-d * (d + 1) / 2)
    for a in range(1, d + 1):
        result *= (1 - 2 ** (d - n - a)) / (1 - 2 ** (-a))
    return result


# Cache the array for performance; not necessary for correctness
@functools.cache
def stabilizer_codim_distribution_cumulative(n):
    """The cumulative distribution function of the
    codimension of the affine subspace of a random
    stabilizer state.

    Parameters
    ----------
    n : int
        Number of qubits in the stabilizer state.

    Returns
    -------
    list[float]
        The dth entry is the probability that the codimension
        is less than or equal to d.
    """

    cumulative = 0
    result = []
    # First compute the unnormalized vector
    for d in range(n + 1):
        cumulative += eta(n, d)
        result.append(cumulative)
    # Now normalize
    for d in range(n + 1):
        result[d] = result[d] / cumulative
    return result


def random_stabilizer_dim(n, rand_gen=None):
    """Sample the dimension of the affine subspace of a
    random stabilizer state.

    Parameters
    ----------
    n : int
        Number of qubits in the stabilizer state.

    Returns
    -------
    int
        A number in [0, n].
    """

    dist = stabilizer_codim_distribution_cumulative(n)
    if rand_gen:
        r = rand_gen.random()
    else:
        r = random.random()  # Uniform on [0, 1]
    for d in range(n + 1):
        if r < dist[d]:
            # Subtract from n to get dimension from codimension
            return n - d


def reduced_row_echelon_and_rank(mat):
    """Compute the reduced row echelon form and rank of a matrix over F2.

    Parameters
    ----------
    mat : array[bool][bool]
        2D numpy array of Booleans, where True = 1 and False = 0 mod 2.

    Returns
    -------
    (array[bool][bool], int)
        The reduced row echelon form of the matrix and its rank.
    """

    mat = mat.copy()
    nr_rows, nr_cols = mat.shape

    # The next row in which we're trying to add a leading 1,
    # which is also a record of the rank so far
    curr_rank = 0
    for col in range(nr_cols):
        if curr_rank == nr_rows:
            # This means we're done, since the rank can't be more than m
            break
        # First try to make mat[curr_rank][col] equal 1
        for row in range(curr_rank + 1, nr_rows):
            if mat[row][col] and not mat[curr_rank][col]:
                # XOR into the goal row
                mat[curr_rank] = np.logical_xor(mat[curr_rank], mat[row])
                break
        # Now zero out the rest of the column, if possible
        if mat[curr_rank][col]:
            for row in range(nr_rows):
                if row != curr_rank and mat[row][col]:
                    mat[row] = np.logical_xor(mat[curr_rank], mat[row])
            curr_rank += 1
    return (mat, curr_rank)


def random_full_rank_reduced(rows, cols, rand_gen=None):
    """Sample a random full-rank matrix over F2 and return its
    reduced row echelon form.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.

    Returns
    -------
    array[bool][bool]
        The reduced row echelon form of a uniformly random full-rank
        matrix over F2.
    """

    goal_rank = min(rows, cols)
    while True:
        # This loop succeeds in each iteration with probability
        # at least 0.288788 in the limit of large dimension,
        # so it should terminate in a reasonable amount of time.
        # TODO maybe put a hard upper bound, just in case?
        if rand_gen:
            mat = rand_gen.randint(2, size=(rows, cols), dtype=bool)
        else:
            mat = np.random.randint(2, size=(rows, cols), dtype=bool)
        reduced_mat, rank = reduced_row_echelon_and_rank(mat)
        if rank == goal_rank:
            return reduced_mat


def stabilizer_gate_list(n):
    """The list of gates that can be toggled on/off to
    synthesize a generic stabilizer state.

    Parameters
    ----------
    n : int
        Number of qubits in the stabilizer state.

    Returns
    -------
    list[str, tup[int]]
        A list of gate names and qubits that the gates act on.
    """

    result = []

    # X layer
    for i in range(n):
        result.append(("x", (i,)))
    # Hadamard layer
    for i in range(n):
        result.append(("h", (i,)))
    # Phase layer
    for i in range(n):
        result.append(("s", (i,)))
    # CZ layer
    for i in range(n):
        for j in range(i + 1, n):
            result.append(("cz", (i, j)))
    # CNOT layer
    for i in range(n):
        for j in range(n):
            if j != i:
                result.append(("cx", (i, j)))
    return result


def random_stabilizer_toggles(n):
    """Generate a random stabilizer state as a list of
    gates to toggle in `stabilizer_gate_list(n)`.

    Parameters
    ----------
    n : int
        Number of qubits in the stabilizer state.

    Returns
    -------
    list[bool]
        A list of the same length as `stabilizer_gate_list(n)`
        indicating which gates to toggle.
    """

    # Sample the dimension of the affine subspace
    dim = random_stabilizer_dim(n)
    # Sample the vector space of the right dimension
    mat = random_full_rank_reduced(dim, n)
    # Hadamard the qubits corresponding to any leading 1
    # The .nonzero()[0][0] reports the first entry that is True
    # Since the matrix is full-rank it will always succeed
    leading_qubits = [row.nonzero()[0][0] for row in mat]

    result = []
    # X layer
    for i in range(n):
        # Apply X gates uniformly at random
        result.append(random.getrandbits(1) == 1)
    # Hadamard layer
    for i in range(n):
        # Apply Hadamard gates to qubits in the set
        result.append(i in leading_qubits)
    # Phase layer
    for i in range(n):
        # Apply S gates uniformly at random,
        # but only to qubits in the set
        result.append(i in leading_qubits and random.getrandbits(1) == 1)
    # CZ layer
    for i in range(n):
        for j in range(i + 1, n):
            # Apply CZ gates uniformly at random,
            # but only to pairs of qubits in the set
            result.append(
                i in leading_qubits
                and j in leading_qubits
                and random.getrandbits(1) == 1
            )
    # CNOT layer
    for i in range(n):
        for j in range(n):
            # Apply CNOT gates between the set and its complement
            # according to the matrix we sampled
            if j != i:
                result.append(
                    i in leading_qubits
                    and j not in leading_qubits
                    and mat[leading_qubits.index(i)][j]
                )
                # leading_qubits.index(i) looks up the row corresponding to i
    return result


def num_stab_gates_ag(n):
    """Number of one- and two-qubit gates needed to synthesize a
    stabilizer state of n qubits via the Aaronson-Gottesman method.
    Equivalent to `len(stabilizer_gate_list_ag(n))`.

    Parameters
    ----------
    n : int
        Number of qubits in the stabilizer state.

    Returns
    -------
    int
        Number of gates in an n-qubit Aaronson-Gottesman circuit
        for stabilizer preparation.
    """

    return n * (n - 1) // 2 + n * 4


def stabilizer_gate_list_ag(n):
    """The list of gates that can be toggled on/off to synthesize
    a generic stabilizer state in Aaronson-Gottesman form. This
    means that we first prepare a graph state and Hadamard some
    of the qubits. Comparable to
    https://docs.quantum.ibm.com/api/qiskit/synthesis#stabilizer-state-synthesis

    Parameters
    ----------
    n : int
        Number of qubits in the stabilizer state.

    Returns
    -------
    list[str, tup[int]]
        A list of gate names and qubits that the gates act on.
    """

    result = []

    # X layer
    for i in range(n):
        result.append(("x", (i,)))
    # Hadamard layer
    for i in range(n):
        result.append(("h", (i,)))
    # Phase layer
    for i in range(n):
        result.append(("s", (i,)))
    # CZ layer
    for i in range(n):
        for j in range(i + 1, n):
            result.append(("cz", (i, j)))
    # Hadamard layer
    for i in range(n):
        result.append(("h", (i,)))
    return result


def random_stabilizer_toggles_ag(n, rand_gen=None):
    """Generate a random stabilizer state as a list of
    gates to toggle in `stabilizer_gate_list_ag(n)`.

    Parameters
    ----------
    n : int
        Number of qubits in the stabilizer state.

    Returns
    -------
    list[bool]
        A list of the same length as `stabilizer_gate_list_ag(n)`
        indicating which gates to toggle.
    """

    # Sample the dimension of the affine subspace
    dim = random_stabilizer_dim(n, rand_gen)
    # Sample the vector space of the right dimension
    mat = random_full_rank_reduced(dim, n, rand_gen)
    # At the end, Hadamard the qubits not corresponding to any leading 1
    # The .nonzero()[0][0] reports the first entry that is True
    # Since the matrix is full-rank it will always succeed
    leading_qubits = [row.nonzero()[0][0] for row in mat]

    result = []
    # X layer
    for i in range(n):
        # Apply X gates uniformly at random
        result.append(random.getrandbits(1) == 1)
    # Hadamard layer
    for i in range(n):
        # Apply Hadamard gates to all qubits
        result.append(True)
    # Phase layer
    for i in range(n):
        # Apply S gates uniformly at random,
        # but only to qubits in the set
        result.append(i in leading_qubits and random.getrandbits(1) == 1)
    # CZ layer
    for i in range(n):
        for j in range(i + 1, n):
            # 3 types of CZ gates:
            # (1) random gates between qubits in S
            # (2) gates where i in S and j not in S, as determined by mat
            # (3) gates where i not in S and j in S, as determined by mat
            # Type (2) and (3) correspond to CNOT gates in random_stabilizer_toggles
            active_1 = (
                i in leading_qubits
                and j in leading_qubits
                and random.getrandbits(1) == 1
            )
            active_2 = (
                i in leading_qubits
                and j not in leading_qubits
                and mat[leading_qubits.index(i)][j]
            )
            # leading_qubits.index(i) looks up the row corresponding to i
            active_3 = (
                i not in leading_qubits
                and j in leading_qubits
                and mat[leading_qubits.index(j)][i]
            )
            # leading_qubits.index(j) looks up the row corresponding to j
            result.append(active_1 or active_2 or active_3)
    # H layer
    for i in range(n):
        # Hadamard the qubits without a leading 1
        result.append(i not in leading_qubits)
    return result
