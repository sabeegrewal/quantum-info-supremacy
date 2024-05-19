import quimb as qu
from ansatz import ansatz_circuit
from optimize import optimize
import numpy as np
import matplotlib.pyplot as plt


def test_random_states_2_to_4():
    threshold = 0.99
    for n in range(2, 4):
        depth = 2 * n
        circ = ansatz_circuit(n, depth)

        psi_dense = qu.rand_haar_state(2**n)
        optimized_unitary, initial_loss, final_loss = optimize(
            circ=circ, target_state_dense=psi_dense
        )
        assert initial_loss != final_loss
        assert final_loss <= 1 - threshold
        print(final_loss)


def plot_averages_and_deviations(x, n, repetitions, mode, *y_lists):
    info = f"(n: {n}, repeat: {repetitions}, mode: {mode})"

    # Assuming all y_lists are of the same length as x and non-empty
    # Stack all y-lists vertically and calculate the mean and standard deviation along columns
    y_stacked = np.vstack(y_lists)
    y_mean = np.mean(y_stacked, axis=0)
    y_std = np.std(y_stacked, axis=0)

    # Plotting the average values with error bars showing the standard deviation
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x,
        y_mean,
        yerr=y_std,
        fmt="o",
        capsize=5,
        linestyle="-",
        color="blue",
        ecolor="blue",
    )

    # Adding labels and title (fill these as needed)
    plt.xlabel("Depth")
    plt.ylabel("Infidelity")
    plt.title(f"Infidelity vs. Depth: {info}")
    plt.grid(True)

    # Save the figure as a PDF
    plt.savefig(f"./plots/plot_{n}_{mode}.pdf")
    plt.show()
    x = 42


def test_plot():
    n = 4
    repetitions = 5
    lists_of_fidelities = []
    x = list(range(1, 2 * n + 1))
    # mode = "random_matching"
    mode = "brickwork"
    for _ in range(repetitions):
        y = []
        psi_dense = qu.rand_haar_state(2**n)
        for depth in x:
            circ = ansatz_circuit(n, depth, two_qubit_mode="brickwork")
            optimized_unitary, initial_loss, final_loss = optimize(
                circ=circ, target_state_dense=psi_dense
            )
            y.append(final_loss)
        lists_of_fidelities.append(y)
    plot_averages_and_deviations(x, n, repetitions, mode, *lists_of_fidelities)


def test_single():
    threshold = 0.01
    n = 9
    depth = 2 * n
    circ = ansatz_circuit(n, depth, two_qubit_mode="brickwork")

    psi_dense = qu.rand_haar_state(2**n)
    optimized_unitary, initial_loss, final_loss = optimize(
        circ=circ, target_state_dense=psi_dense
    )
    assert initial_loss != final_loss
    assert final_loss <= threshold
    # print(final_loss)
