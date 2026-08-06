import math
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit.transpiler import CouplingMap
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import UnitaryGate


def c_amodN(a: int, power: int, N: int) -> QuantumCircuit:
    n = math.ceil(math.log2(N + 1))
    val = pow(a, 2 ** power, N)

    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)

    for x in range(dim):
        if x < N:
            target_x = (val * x) % N
        else:
            target_x = x

        U[target_x, x] = 1.0

    gate = UnitaryGate(U, label=f"U_({a}^{2 ** power} mod {N})")
    qc = QuantumCircuit(n)
    qc.append(gate, range(n))
    return qc.control(1)


def qft_dagger(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)
        qc.h(j)
    qc.name = "QFT_dagger"
    return qc


def build_shor_circuit(N: int, a: int) -> QuantumCircuit:
    n_count = math.ceil(math.log2(N + 1))
    qr_up = QuantumRegister(n_count, name='up')
    qr_down = QuantumRegister(n_count, name='down')
    cr = ClassicalRegister(n_count, name='meas')
    qc = QuantumCircuit(qr_up, qr_down, cr)

    for q in range(n_count):
        qc.h(qr_up[q])

    qc.x(qr_down[0])

    for q in range(n_count):
        qc.append(c_amodN(a, q, N), [qr_up[q]] + qr_down[:])

    qc.append(qft_dagger(n_count), qr_up)
    qc.measure(qr_up, cr)

    return qc


def create_synthetic_odra5_noise():
    nm = NoiseModel()
    p1 = 0.001
    p2 = 0.012
    p_ro = 0.018

    for q in range(5):
        ro_err = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
        nm.add_readout_error(ro_err, [q])
        g_err = depolarizing_error(p1, 1)
        nm.add_quantum_error(g_err, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'], [q])

    edges = [[0, 2], [1, 2], [2, 3], [2, 4], [2, 0], [2, 1], [3, 2], [4, 2]]
    for edge in edges:
        g2_err = depolarizing_error(p2, 2)
        nm.add_quantum_error(g2_err, ['cx', 'cz'], edge)

    return nm


def extend_odra_noise_model(base_noise_model: NoiseModel, base_coupling_list: list, n_qubits: int, base_n: int = 5):
    """Skaluje model szumu Odry 5 do wymaganej liczby kubitów."""
    if n_qubits <= base_n:
        return base_noise_model, base_coupling_list

    extended_nm = copy.deepcopy(base_noise_model)
    extended_coupling = list(base_coupling_list)

    donor_map = {q: random.randint(0, base_n - 1) for q in range(base_n, n_qubits)}

    for new_q, donor_q in donor_map.items():
        hub = 2 if 2 < base_n else donor_q
        extended_coupling.append([new_q, hub])
        extended_coupling.append([hub, new_q])

    if hasattr(base_noise_model, '_readout_errors'):
        for q_tuple, ro_err in base_noise_model._readout_errors.items():
            for new_q, donor_q in donor_map.items():
                if q_tuple == (donor_q,) or q_tuple == donor_q:
                    extended_nm.add_readout_error(ro_err, [new_q])

    if hasattr(base_noise_model, '_local_quantum_errors'):
        for op, q_dict in base_noise_model._local_quantum_errors.items():
            if isinstance(q_dict, dict):
                for q_tuple, qerror in q_dict.items():
                    if len(q_tuple) == 1:
                        src_q = q_tuple[0]
                        for new_q, donor_q in donor_map.items():
                            if src_q == donor_q:
                                extended_nm.add_quantum_error(qerror, op, [new_q])
                    elif len(q_tuple) == 2:
                        q1, q2 = q_tuple
                        hub = 2 if base_n > 2 else 0
                        for new_q, donor_q in donor_map.items():
                            target_donor = donor_q if donor_q != hub else 0
                            if (q1, q2) == (target_donor, hub):
                                extended_nm.add_quantum_error(qerror, op, [new_q, hub])
                            elif (q1, q2) == (hub, target_donor):
                                extended_nm.add_quantum_error(qerror, op, [hub, new_q])

    return extended_nm, extended_coupling


def get_odra5_backend(n_qubits: int = 8):
    noise_model = create_synthetic_odra5_noise()
    base_coupling_list = [[0, 2], [1, 2], [2, 3], [2, 4], [2, 0], [2, 1], [3, 2], [4, 2]]

    extended_nm, extended_coupling_list = extend_odra_noise_model(
        noise_model, base_coupling_list, n_qubits=n_qubits, base_n=5
    )

    coupling_map = CouplingMap(extended_coupling_list)
    noisy_sim = AerSimulator(noise_model=extended_nm)
    noisy_sim.set_options(noise_model=extended_nm, coupling_map=coupling_map)

    return noisy_sim


def analyze_quantum_counts(counts: dict, N: int, n_count: int, a: int, label: str):
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for bitstring, count in sorted_counts:
        meas_int = int(bitstring, 2)
        if meas_int == 0:
            continue

        phase = meas_int / (2 ** n_count)
        frac = Fraction(phase).limit_denominator(N)
        candidate_r = frac.denominator

        if candidate_r % 2 != 0:
            continue

        if pow(a, candidate_r // 2, N) == (N - 1):
            continue

        g1 = math.gcd(pow(a, candidate_r // 2, N) - 1, N)
        g2 = math.gcd(pow(a, candidate_r // 2, N) + 1, N)

        if g1 not in [1, N] and g2 not in [1, N]:
            p_val = min(g1, g2)
            q_val = max(g1, g2)
            print(f"r = {candidate_r}")
            print(f"p = {p_val}, q = {q_val}")
            return p_val, q_val, candidate_r

    print(f"Fail - no valid r found in {label} counts.")
    return None, None, None


def factorize_N_quantum(N: int, noisy: bool = True):
    n_count = math.ceil(math.log2(N + 1))
    sim_ideal = AerSimulator()
    sim_noisy = get_odra5_backend(n_qubits=2 * n_count)

    attempt = 1
    tested_a = set()

    while True:
        print(f"{attempt}.")

        available_a = [candidate for candidate in range(2, N) if candidate not in tested_a]
        if not available_a:
            print("[Error] Every a tried.")
            return None, None

        a = random.choice(available_a)
        tested_a.add(a)
        print(f"a = {a}")

        gcd_val = math.gcd(a, N)
        if gcd_val > 1:
            p_found = min(gcd_val, N // gcd_val)
            q_found = max(gcd_val, N // gcd_val)
            print(f"Luck: NWD({a}, {N}) = {gcd_val} > 1 (no quantum needed)")
            print(f"N={N}: p = {p_found}, q = {q_found}")
            return p_found, q_found

        circuit = build_shor_circuit(N=N, a=a)


        p_ideal,q_ideal, p_noisy, q_noisy = None, None, None, None

        if not noisy:
            print("Ideal try")
            transpiled_ideal = transpile(circuit, sim_ideal)
            counts_ideal = sim_ideal.run(transpiled_ideal, shots=1024).result().get_counts()

            p_ideal, q_ideal, _ = analyze_quantum_counts(counts_ideal, N, n_count, a, "SYMULATOR IDEALNY")
        if noisy:
            print("Noisy try")
            # transpiled_noisy = transpile(circuit, sim_noisy)
            # counts_noisy = sim_noisy.run(transpiled_noisy, shots=1024).result().get_counts()
            # print(counts_noisy)
            counts_noisy = {'1000': 470, '0000': 426, '1010': 11, '0001': 12, '0101': 5, '1110': 9, '0010': 20, '1100': 23, '0100': 25, '0110': 7, '1001': 12, '1011': 1, '1101': 3}

            p_noisy, q_noisy, _ = analyze_quantum_counts(counts_noisy, N, n_count, a, "SZUMOWY (ODRA 5)")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        if not noisy:
            plot_histogram(counts_ideal, ax=ax1, color='midnightblue', title=f"Idealny Symulator (Próba #{attempt}, a={a})")
        if noisy:
            plot_histogram(counts_noisy, ax=ax2, color='crimson', title=f"Odra 5 z Szumem (Próba #{attempt}, a={a})")
        plt.tight_layout()
        plt.show()

        if p_ideal or p_noisy:
            p_final = p_ideal if p_ideal else p_noisy
            q_final = q_ideal if q_ideal else q_noisy

            print(f"N={N}: p = {p_final}, q = {q_final}")
            print(f"{p_final} * {q_final} = {p_final * q_final} ({p_final * q_final == N})")
            return p_final, q_final

        print(f"Fail")
        attempt += 1


if __name__ == "__main__":
    N_to_factor = 15
    p_res, q_res = factorize_N_quantum(N=N_to_factor)