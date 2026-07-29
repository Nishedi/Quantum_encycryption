import os
import math
import random
import copy
import requests
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error
)
from qiskit.transpiler import CouplingMap
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import UnitaryGate


def prepare_rsa():
    p, q = 3, 11
    N = p * q  # N = 15
    phi = (p - 1) * (q - 1)
    e = 3
    d = pow(e, -1, phi)
    return N, e, d, p, q


def encrypt_rsa(msg: int, e: int, N: int) -> int:
    return pow(msg, e, N)


def decrypt_rsa(ciphertext: int, d: int, N: int) -> int:
    return pow(ciphertext, d, N)


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
    print(f"\n--- Analiza wyników: {label} ---")
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
            print(f"  [SUKCES {label}] Zmierzono {bitstring} ({count} shotów) -> Okres r = {candidate_r} -> p = {g1}, q = {g2}")
            return g1, g2, candidate_r

    print(f"  [PORAŻKA {label}] Żaden z pików nie dał poprawnych czynników pierwszych.")
    return None, None, None


def run_shor_attack_dual(N: int, ciphertext: int, e: int, msg_original: int):
    n_count = math.ceil(math.log2(N + 1))
    sim_ideal = AerSimulator()
    sim_noisy = get_odra5_backend(n_qubits=2 * n_count)

    attempt = 1
    tested_a = set()


    while True:

        available_a = [candidate for candidate in range(2, N) if candidate not in tested_a]
        if not available_a:
            print("[BŁĄD] Wypróbowano wszystkie możliwe wartości 'a'.")
            break

        a = random.choice(available_a)
        tested_a.add(a)
        print(f"a = {a}")

        gcd_val = math.gcd(a, N)
        if gcd_val > 1:
            print(f"NWD({a}, {N}) = {gcd_val} > 1!")
            print(f"Czynniki znalezione klasycznie bez komputera kwantowego: p = {gcd_val}, q = {N // gcd_val}")
            break

        print(f" NWD({a}, {N}) = 1. Uruchamiamy obwód kwantowy na obu symulatorach...")

        circuit = build_shor_circuit(N=N, a=a)

        transpiled_ideal = transpile(circuit, sim_ideal)
        counts_ideal = sim_ideal.run(transpiled_ideal, shots=1024).result().get_counts()

        transpiled_noisy = transpile(circuit, sim_noisy)
        counts_noisy = sim_noisy.run(transpiled_noisy, shots=1024).result().get_counts()

        p_ideal, q_ideal, r_ideal = analyze_quantum_counts(counts_ideal, N, n_count, a, "SYMULATOR IDEALNY")
        p_noisy, q_noisy, r_noisy = analyze_quantum_counts(counts_noisy, N, n_count, a, "SZUMOWY (ODRA 5)")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        plot_histogram(counts_ideal, ax=ax1, color='midnightblue', title=f"Idealny Symulator (Próba #{attempt}, a={a})")
        plot_histogram(counts_noisy, ax=ax2, color='crimson', title=f"FakeOdraRealTime z Szumem (Próba #{attempt}, a={a})")
        plt.tight_layout()
        # plt.show()

        if p_ideal:
            p_final = p_ideal
            q_final = q_ideal


            phi_cracked = (p_final - 1) * (q_final - 1)
            d_cracked = pow(e, -1, phi_cracked)
            decrypted_msg = decrypt_rsa(ciphertext, d_cracked, N)

            print(f"\n[WYNIKI DESZYFROWANIA RSA]")
            print(f" • Moduł N = {N} (p={p_final}, q={q_final})")
            print(f" • Odzyskany klucz prywatny d = {d_cracked}")
            print(f" • Zaszyfrowany kryptogram C = {ciphertext}")
            print(f" • Odszyfrowana wiadomość M = {decrypted_msg}")
            print(f" • Poprawność deszyfrowania: {decrypted_msg == msg_original}")

        if p_noisy:
            p_final = p_noisy
            q_final = q_noisy

            print(f" SUKCES ATAKU W PROBIE #{attempt} (dla a = {a})! i symulatora z szumami")

            phi_cracked = (p_final - 1) * (q_final - 1)
            d_cracked = pow(e, -1, phi_cracked)
            decrypted_msg = decrypt_rsa(ciphertext, d_cracked, N)

            print(f"\n[WYNIKI DESZYFROWANIA RSA]")
            print(f" • Moduł N = {N} (p={p_final}, q={q_final})")
            print(f" • Odzyskany klucz prywatny d = {d_cracked}")
            print(f" • Zaszyfrowany kryptogram C = {ciphertext}")
            print(f" • Odszyfrowana wiadomość M = {decrypted_msg}")
            print(f" • Poprawność deszyfrowania: {decrypted_msg == msg_original}")

        if p_ideal or p_noisy:
            break

        print(f"[KLASYCZNIE] Próba #{attempt} nie dała poprawnych czynników. Losuję inne 'a'...")
        attempt += 1


def main():
    M = 13
    N, e, d_real, p_real, q_real = prepare_rsa()
    ciphertext = encrypt_rsa(M, e, N)

    print(f"• Parametry RSA: N = {N} (p={p_real}, q={q_real}) | e = {e} | d_tajne = {d_real}")
    print(f"• Tajna wiadomość M = {M}")
    print(f"• Zaszyfrowany kryptogram C = {ciphertext}")

    run_shor_attack_dual(N=N, ciphertext=ciphertext, e=e, msg_original=M)


if __name__ == "__main__":
    main()