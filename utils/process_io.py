from pytket import Bit

import time


def await_job(backend, result_handle, sleep=20, trials=1000):
    """Retrieve a job from the server, waiting until completed.

    Parameters
    ----------
    backend : Backend
        Backend to submit the job to.
    result_handle : ResultHandle
        Result handle for the job.
    sleep : real
        Optional time in seconds to wait between intervals of checking whether the job is finished.
        Defaults to 20.
    trials : int
        Number of trials to attempt getting the results.
        Defaults to 1000.

    Returns
    -------
    BackendResult
        The BackendResult obtained either when the job is completed,
        errored, cancelled, or the number of trials runs out.
    """

    for trial in range(1000):
        result, job_status = backend.get_partial_result(result_handle)
        if job_status.status.name == "COMPLETED":
            print("Done!")
            break
        elif job_status.status.name == "ERROR":
            print("ERROR")
            break
        elif job_status.status.name == "CANCELLED":
            print("CANCELLED")
            break
        else:
            print("Waiting...")
        time.sleep(sleep)
    return result


def get_shots_and_xeb_scores(scoring_states, detect_leakage, result):
    """Get the shots and compute the XEB scores from the results on the
    given scoring states.

    Parameters
    ----------
    scoring_states : array
        List of complex arrays of shape `[2] * n' for some `n`.
    detect_leakage : bool
        Whether to the leakage detection gadget was used.
    result : BackendResult
        Result containing samples from the output distribution.

    Returns
    -------
    list[list[(str, real)]]
        A list of lists of pairs of (shot, XEB score) corresponding to the
        shots from each state, with pruning of all shots detected as leaky.
    """

    n = len(scoring_states[0].shape)

    all_scores = []
    for circ_idx in range(len(scoring_states)):
        # The bits that were measured in each shot
        measured_bits = [Bit(f"measurement{circ_idx}", i) for i in range(n)]
        if detect_leakage:
            measured_bits = measured_bits + [
                Bit(f"leakage_detection{circ_idx}", i) for i in range(n)
            ]
        all_shots_with_leakage_results = result.get_shots(cbits=measured_bits)
        
        # Prunes all shots detected as leaky
        pruned_shots = [
            shot[:n] for shot in all_shots_with_leakage_results if not any(shot[n:])
        ]
        
        pruned_shots_and_xeb_scores = [
            (
                "".join(str(bit) for bit in shot),
                abs(scoring_states[circ_idx][tuple(shot)]) ** 2 * 2**n - 1
            )
            for shot in pruned_shots
        ]

        all_scores.append(pruned_shots_and_xeb_scores)
    return all_scores
