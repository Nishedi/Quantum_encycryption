import math
import random
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import UnitaryGate
from simulators import get_iqm_backend


def c_amodN(a: int, power: int, N: int) -> QuantumCircuit:
    """
    Tworzy bramkę kontrolowanego potęgowania modularnego: U |x> = |a^(2^power) * x mod N>.
    """
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
    """
    Konstruuje obwód Shora wykorzystujący Iterative Quantum Phase Estimation (IQPE).
    Zamiast używać rejestru zliczającego o rozmiarze n_count, używa TYLKO 1 kubitu
    oraz dynamicznego pomiaru i resetu (Dynamic Circuits).
    """
    n_count = math.ceil(math.log2(N + 1))

    # Potrzebujemy 1 kubitu zliczającego i n_count kubitów roboczych
    qr_meas = QuantumRegister(1, name='q_meas')
    qr_work = QuantumRegister(n_count, name='q_work')
    # Rejestr klasyczny na n_count bitów, do którego będziemy zapisywać kolejne pomiary
    cr = ClassicalRegister(n_count, name='c_meas')

    qc = QuantumCircuit(qr_meas, qr_work, cr)

    # Inicjalizacja rejestru roboczego na stan |1>
    qc.x(qr_work[0])

    # Iteracyjne szacowanie fazy (od najmniej znaczącego bitu do najbardziej)
    # Zauważ: iterujemy od tyłu (n_count-1 w dół do 0)
    for i in range(n_count - 1, -1, -1):
        # 1. Reset kubitu pomiarowego do |0> (konieczne przy pętli)
        qc.reset(qr_meas)
        qc.h(qr_meas)

        # 2. Kontrolowane potęgowanie modularne (zawsze z wykładnikiem zależnym od i)
        # Zależność od 'i': wykonujemy a^(2^i) mod N
        c_U = c_amodN(a, i, N)
        qc.append(c_U, [qr_meas[0]] + qr_work[:])

        # 3. Klasyczne sprzężenie zwrotne (Classical Feedback)
        # Obracamy fazę na podstawie wcześniej zmierzonych bitów
        for j in range(n_count - 1, i, -1):
            # Jeśli bit c_meas[j] był 1, wykonujemy obrót fazy
            # qiskit obsługuje c_if na całym rejestrze lub pojedynczym bicie
            # Dla IQPE kąt obrotu to -pi / 2^(j - i)
            angle = -np.pi / (2 ** (j - i))
            qc.p(angle, qr_meas[0]).c_if(cr[j], 1)

        # 4. Bramka H i pomiar do bitu 'i'
        qc.h(qr_meas)
        qc.measure(qr_meas[0], cr[i])

    return qc


def analyze_iqpe_counts(counts: dict, N: int, n_count: int, a: int, label: str):
    """
    Analizuje wyniki zliczeń ze zredukowanego obwodu IQPE.
    Logika ułamków ciągłych jest identyczna jak w standardowym QFT.
    """
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for bitstring, count in sorted_counts:
        # bitstring jest czytany bezpośrednio z rejestru cr
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


def factorize_N_iqpe(N: int, noisy: bool = True):
    """
    Główna pętla faktoryzacji korzystająca z IQPE i fizycznego modelu szumu IQM.
    """
    n_count = math.ceil(math.log2(N + 1))
    sim_ideal = AerSimulator()
    sim_noisy = get_iqm_backend("calibration_data.json")
    attempt = 1
    tested_a = set()

    while True:
        print(f"\n[IQPE] Attempt {attempt}.")

        available_a = [candidate for candidate in range(2, N) if candidate not in tested_a]
        if not available_a:
            print("[Error] Every 'a' tried.")
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

        circuit = build_iqpe_shor_circuit(N=N, a=a)

        print(f"Zbudowano obwód IQPE. Wymaga on zaledwie {circuit.num_qubits} kubitów (zamiast {2 * n_count}).")

        p_ideal, q_ideal, p_noisy, q_noisy = None, None, None, None

        if not noisy:
            print("Ideal try")
            transpiled_ideal = transpile(circuit, sim_ideal)
            counts_ideal = sim_ideal.run(transpiled_ideal, shots=1024).result().get_counts()
            p_ideal, q_ideal, _ = analyze_iqpe_counts(counts_ideal, N, n_count, a, "SYMULATOR IDEALNY")

        if noisy:
            print("Noisy try (Optimization Level 3)")
            # Maksymalna kompresja, kluczowa przy długich obwodach w NISQ
            transpiled_noisy = transpile(circuit, sim_noisy, optimization_level=3)
            counts_noisy = sim_noisy.run(transpiled_noisy, shots=1024).result().get_counts()
            p_noisy, q_noisy, _ = analyze_iqpe_counts(counts_noisy, N, n_count, a, "SZUMOWY (IQM Sirius)")

        if p_ideal or p_noisy:
            p_final = p_ideal if p_ideal else p_noisy
            q_final = q_ideal if q_ideal else q_noisy

            print(f"\nSUCCESS! N={N}: p = {p_final}, q = {q_final}")
            print(f"{p_final} * {q_final} = {p_final * q_final} ({p_final * q_final == N})")
            return p_final, q_final

        print(f"Fail - retrying with new 'a'")
        attempt += 1


if __name__ == "__main__":
    # Testujemy większy moduł N=35
    ps = [3]
    qs = [5]
    for p in ps:
        for q in qs:
            if p == q:
                continue
            N = p * q
            print(f"Trying to factor N={N} (p={p}, q={q}) using IQPE")

            p_res, q_res = factorize_N_iqpe(N=N, noisy=True)