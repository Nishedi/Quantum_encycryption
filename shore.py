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
from simulators import get_iqm_backend, get_sirius_real_backend
import time


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


def factorize_N_quantum(N: int, ideal: bool = False, real_sirius: bool = False, noisy: bool = False, a_set = None):
    n_count = math.ceil(math.log2(N + 1))
    sim_ideal = AerSimulator()
    # sim_noisy = get_odra5_backend(n_qubits=2 * n_count)
    sim_noisy = get_iqm_backend("calibration_data.json")
    attempt = 1
    tested_a = set()

    while True:
        print(f"{attempt}.")

        available_a = [candidate for candidate in range(2, N) if candidate not in tested_a]
        if not available_a:
            print("[Error] Every a tried.")
            return None, None
        if a_set is None:
            a = random.choice(available_a)
            tested_a.add(a)
        else:
            a = a_set
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

        if ideal:
            print("Ideal try")
            transpiled_ideal = transpile(circuit, sim_ideal)
            counts_ideal = sim_ideal.run(transpiled_ideal, shots=1024).result().get_counts()
            p_ideal, q_ideal, _ = analyze_quantum_counts(counts_ideal, N, n_count, a, "SYMULATOR IDEALNY")
        single_q_gates = 0
        two_q_gates = 0
        total_gates = 0
        circuit_duration_dt = None
        circuit_depth = 0
        if noisy:
            print("Noisy try")
            transpilation_time_start = time.time()
            transpiled_noisy = transpile(circuit, sim_noisy)
            transpilation_time = time.time() - transpilation_time_start
            circuit_depth = transpiled_noisy.depth()
            for instruction in transpiled_noisy.data:
                op = instruction.operation if hasattr(instruction, 'operation') else instruction[0]

                if op.name not in ['measure', 'barrier', 'delay']:
                    total_gates += 1
                    if op.num_qubits == 1:
                        single_q_gates += 1
                    elif op.num_qubits == 2:
                        two_q_gates += 1

            circuit_duration_dt = transpiled_noisy.duration

            # return
            run_time_start = time.time()
            counts_noisy = sim_noisy.run(transpiled_noisy, shots=1024).result().get_counts()
            run_time = time.time() - run_time_start
            print(f"Transpilation time: {transpilation_time:.4f} seconds, Run time: {run_time:.4f} seconds")

            p_noisy, q_noisy, _ = analyze_quantum_counts(counts_noisy, N, n_count, a, "SZUMOWY (ODRA 5)")

        if real_sirius:
            real_sirius_device = get_sirius_real_backend()
            transpiled_real_sirius = transpile(circuit, real_sirius_device)
            circuit_depth = transpiled_real_sirius.depth()
            for instruction in transpiled_real_sirius.data:
                op = instruction.operation if hasattr(instruction, 'operation') else instruction[0]

                if op.name not in ['measure', 'barrier', 'delay']:
                    total_gates += 1
                    if op.num_qubits == 1:
                        single_q_gates += 1
                    elif op.num_qubits == 2:
                        two_q_gates += 1

            circuit_duration_dt = transpiled_real_sirius.duration



        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        if ideal:
            plot_histogram(counts_ideal, ax=ax1, color='midnightblue', title=f"Idealny Symulator (Próba #{attempt}, a={a})")
        if noisy:
            plot_histogram(counts_noisy, ax=ax2, color='crimson', title=f"Odra 5 z Szumem (Próba #{attempt}, a={a})")
        plt.tight_layout()
        # plt.show()

        if p_ideal or p_noisy:
            p_final = p_ideal if p_ideal else p_noisy
            q_final = q_ideal if q_ideal else q_noisy

            print(f"N={N}: p = {p_final}, q = {q_final}")
            print(f"{p_final} * {q_final} = {p_final * q_final} ({p_final * q_final == N})")

            with open ("circuit_metrics.csv", "a") as f:
                f.write(f"{N},{a},FakeSirius,{circuit_depth},{total_gates},{single_q_gates},{two_q_gates}")
                if circuit_duration_dt is not None:
                    f.write(f",{circuit_duration_dt}\n")
                else:
                    f.write(",None\n")

            return p_final, q_final
        if real_sirius:
            with open ("circuit_metrics.csv", "a") as f:
                f.write(f"{N},{a},RealSirius,{circuit_depth},{total_gates},{single_q_gates},{two_q_gates}")
                if circuit_duration_dt is not None:
                    f.write(f",{circuit_duration_dt}\n")
                else:
                    f.write(",None\n")
        if a_set:
            return -1, -1
        print(f"Fail")
        attempt += 1


if __name__ == "__main__":
    ps = [5]
    qs = [7]
    repeats = [2,3,4,6,8,10,11,13,14,16,17,18,19,20,21,22,23,24,26,27,28,29,31,32,33,34]

    for p in ps:
        for q in qs:
            if p == q:
                continue
            N = p * q
            print(f"Trying to factor N={N} (p={p}, q={q})")
            for r in repeats:
                p_res, q_res = factorize_N_quantum(N=N, a_set=r, real_sirius=True)
            # for r in repeats:
            #     p_res, q_res = factorize_N_quantum(N=N, a_set=r, real_sirius=True)