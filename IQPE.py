import math
import random
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import time

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import UnitaryGate
from simulators import get_iqm_backend, get_real_iqm_backend, get_sirius_real_backend





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














def build_iqpe_shor_circuit(N: int, a: int) -> QuantumCircuit:
    n_count = math.ceil(math.log2(N + 1))

    qr_meas = QuantumRegister(1, name='q_meas')
    qr_work = QuantumRegister(n_count, name='q_work')
    cr = ClassicalRegister(n_count, name='c_meas')

    qc = QuantumCircuit(qr_meas, qr_work, cr)

    qc.x(qr_work[0])

    for i in range(n_count - 1, -1, -1):
        qc.reset(qr_meas)
        qc.h(qr_meas)

        c_U = c_amodN(a, i, N)
        qc.append(c_U, [qr_meas[0]] + qr_work[:])

        for j in range(n_count - 1, i, -1):
            angle = -np.pi / (2 ** (j - i))
            qc.p(angle, qr_meas[0]).c_if(cr[j], 1)


        qc.h(qr_meas)
        qc.measure(qr_meas[0], cr[i])

    return qc


def analyze_iqpe_counts(counts: dict, N: int, n_count: int, a: int, label: str, verbose: bool = True):
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
            if verbose:
                print(f"r = {candidate_r}")
                print(f"p = {p_val}, q = {q_val}")
            return p_val, q_val, candidate_r
    if verbose:
        print(f"Fail - no valid r found in {label} counts.")
    return None, None, None


def factorize_N_iqpe(N: int, ideal: bool = False, noisy: bool = False, real_odra: bool = False, real_sirius: bool = False, a: int = None, verbose: bool = True):
    random_a = False if a else True
    n_count = math.ceil(math.log2(N + 1))
    sim_ideal = AerSimulator()
    sim_noisy = get_iqm_backend("calibration_data.json")
    if real_odra:
        real_odra_device = get_real_iqm_backend()
    if real_sirius:
        real_sirius_device = get_sirius_real_backend()
    attempt = 1
    tested_a = set()

    while True:
        if verbose:
            print(f"\n[IQPE] Attempt {attempt}.")

        available_a = [candidate for candidate in range(2, N) if candidate not in tested_a]
        if not available_a:
            if verbose:
                print("[Error] Every 'a' tried.")
            return None, None, None

        if random_a:
            a = random.choice(available_a)
            print(f"Randomly selected a = {a}")


        tested_a.add(a)
        if verbose:
            print(f"a = {a}")

        gcd_val = math.gcd(a, N)
        if gcd_val > 1:
            p_found = min(gcd_val, N // gcd_val)
            q_found = max(gcd_val, N // gcd_val)
            if verbose:
                print(f"Luck: NWD({a}, {N}) = {gcd_val} > 1 (no quantum needed)")
                print(f"N={N}: p = {p_found}, q = {q_found}")
            return p_found, q_found, None

        circuit = build_iqpe_shor_circuit(N=N, a=a)
        if verbose:
            print(f"IQPE circuit builded - {circuit.num_qubits} required.")

        p_ideal, q_ideal, p_noisy, q_noisy, p_real_odra, q_real_odra, p_real_sirius, q_real_sirius = None, None, None, None, None, None, None, None

        if ideal:
            transpiled_ideal = transpile(circuit, sim_ideal)
            counts_ideal = sim_ideal.run(transpiled_ideal, shots=1024).result().get_counts()
            p_ideal, q_ideal, _ = analyze_iqpe_counts(counts_ideal, N, n_count, a, "SYMULATOR IDEALNY", verbose=verbose)

        if noisy:
            transpiled_noisy = transpile(circuit, sim_noisy, optimization_level=3)
            start_time = time.time()
            counts_noisy = sim_noisy.run(transpiled_noisy, shots=1024).result().get_counts()
            end_time = time.time()
            p_noisy, q_noisy, _ = analyze_iqpe_counts(counts_noisy, N, n_count, a, "SZUMOWY (IQM Sirius)", verbose=verbose)

        if real_odra:
            transpiled_real_odra = transpile(circuit, real_odra_device, optimization_level=3)
            start_time = time.time()
            counts_real_odra = real_odra_device.run(transpiled_real_odra, shots=1024).result().get_counts()
            end_time = time.time()
            p_real_odra, q_real_odra, _ = analyze_iqpe_counts(counts_real_odra, N, n_count, a, "REALNY (IQM Odra5)", verbose=verbose)

        if real_sirius:
            transpiled_real_sirius = transpile(circuit, real_sirius_device, optimization_level=3)
            print("Starting computation on QPU...")
            start_time = time.time()
            counts_real_sirius = real_sirius_device.run(transpiled_real_sirius, shots=128).result().get_counts()
            end_time = time.time()
            print("Computation finished.")
            p_real_sirius, q_real_sirius, _ = analyze_iqpe_counts(counts_real_sirius, N, n_count, a, "REALNY (IQM Sirius)", verbose=verbose)

        if p_ideal or p_noisy or p_real_odra or p_real_sirius:
            p_final = p_ideal if p_ideal else p_noisy
            q_final = q_ideal if q_ideal else q_noisy
            if real_odra and p_real_odra:
                p_final = p_real_odra
                q_final = q_real_odra

            if real_sirius and p_real_sirius:
                p_final = p_real_sirius
                q_final = q_real_sirius
            if verbose:
                print(f"\nSUCCESS! N={N}: p = {p_final}, q = {q_final}")
                print(f"{p_final} * {q_final} = {p_final * q_final} ({p_final * q_final == N}), time {end_time - start_time:.2f}s")
            return p_final, q_final, end_time - start_time

        if verbose:
            print(f"Fail - retrying with new 'a' time {end_time - start_time:.2f}s")
        if random_a:
            attempt += 1
        else:
            return None, None, None


if __name__ == "__main__":
    ps = [3]
    qs = [7]
    a = 2
    repeats = 1
    for p in ps:
        for q in qs:
            if p == q:
                continue
            N = p * q
            for _ in range(repeats):
                # print(f"Trying to factor N={N} (p={p}, q={q}, a={a}) using IQPE")

                p_res, q_res, r_time = factorize_N_iqpe(N=N, real_sirius=True,a=a, verbose=False)
                if p_res is not None and q_res is not None and r_time is not None:
                    print(f"Result: p = {p_res}, q = {q_res}, time = {r_time:.2f}s")
                else:
                    print(f"Failed to factor N for p={p}, q={q}, a={a}.")



# N = 15 correct_a = [2, 7, 8, 13]
# N = 21 correct_a =  [2, 10, 11, 13, 19]
# N = 33 incorrect_a = 2,3

# SIRIUS N = 15  5.76s 7.30