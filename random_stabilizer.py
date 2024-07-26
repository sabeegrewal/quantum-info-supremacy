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
    
    result = 2**(-d*(d+1)/2)
    for a in range(1, d+1):
        result *= (1 - 2**(d-n-a)) / (1 - 2**(-a))
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
    for d in range(n+1):
        cumulative += eta(n, d)
        result.append(cumulative)
    # Now normalize
    for d in range(n+1):
        result[d] = result[d] / cumulative
    return result

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

def random_full_rank_reduced(rows, cols):
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
        mat = np.random.randint(2, size=(rows, cols), dtype=bool)
        reduced_mat, rank = reduced_row_echelon_and_rank(mat)
        if rank == goal_rank:
            return reduced_mat

def random_stabilizer_dim(n):
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
    r = random.random() # Uniform on [0, 1]
    for d in range(n+1):
        if r < dist[d]:
            # Subtract from n to get dimension from codimension 
            return n - d

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
        for j in range(i+1, n):
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

    result = []

    # X layer
    for i in range(n):
        # Apply X gates uniformly at random
        result.append(random.getrandbits(1) == 1)

    # Sample the dimension of the affine subspace
    dim = random_stabilizer_dim(n)
    # Now sample the qubits to Hadamard
    # This is a random subset of size dim
    hadamard_qubits = list(range(n))
    random.shuffle(hadamard_qubits)
    hadamard_qubits = set(hadamard_qubits[:dim])
    
    # Hadamard layer
    for i in range(n):
        # Apply Hadamard gates to qubits in the set
        result.append(i in hadamard_qubits)
    # Phase layer
    for i in range(n):
        # Apply S gates uniformly at random,
        # but only to qubits in the set
        result.append(i in hadamard_qubits
                      and random.getrandbits(1) == 1)
    # CZ layer
    for i in range(n):
        for j in range(i+1, n):
            # Apply CZ gates uniformly at random,
            # but only to pairs of qubits in the set
            result.append(i in hadamard_qubits and
                          j in hadamard_qubits and
                          random.getrandbits(1) == 1)
    # CNOT layer
    for i in range(n):
        for j in range(n):
            # Apply Hadamard gates uniformly at random,
            # but only between the set and its complement
            if j != i:
                result.append(i in hadamard_qubits and
                              j not in hadamard_qubits and
                              random.getrandbits(1) == 1)
    return result
