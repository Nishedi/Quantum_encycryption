import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit.visualization import plot_histogram


def create_synthetic_odra5_noise():
    """Generuje syntetyczny model szumu procesora Odra 5 dla 3 kubitów."""
    nm = NoiseModel()
    p1 = 0.001
    p2 = 0.012
    p_ro = 0.018

    for q in range(3):
        ro_err = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
        nm.add_readout_error(ro_err, [q])
        g_err = depolarizing_error(p1, 1)
        nm.add_quantum_error(g_err, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'], [q])

    edges = [[0, 1], [1, 2], [1, 0], [2, 1]]
    for edge in edges:
        g2_err = depolarizing_error(p2, 2)
        nm.add_quantum_error(g2_err, ['cx', 'cz'], edge)

    return nm


def classical_factor_cost(x1: int, y1: int, y2: int, N: int = 15) -> float:
    """
    Funkcja celu VQF dla N = 15.
    Liczby nieparzyste reprezentujemy jako:
      p = 1 + 2*x1
      q = 1 + 2*y1 + 4*y2
    Koszt = (p * q - N)^2
    """
    p = 1 + 2 * x1
    q = 1 + 2 * y1 + 4 * y2
    return (p * q - N) ** 2


def build_vqf_ansatz(params: np.ndarray, num_qubits: int = 3, reps: int = 1) -> QuantumCircuit:
    """
    Tworzy obwód wariacyjny (Hardware-Efficient Ansatz) na 3 kubitach.
    Kubit 0: x1 (dla p)
    Kubit 1: y1 (dla q)
    Kubit 2: y2 (dla q)
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    param_idx = 0

    # Warstwa wstępnych obrotów Ry
    for q in range(num_qubits):
        qc.ry(params[param_idx], q)
        param_idx += 1

    # Warstwy powtarzalne (Splot CNOT + Obroty)
    for _ in range(reps):
        qc.cx(0, 1)
        qc.cx(1, 2)
        for q in range(num_qubits):
            qc.ry(params[param_idx], q)
            param_idx += 1

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def compute_energy(params: np.ndarray, simulator, N: int = 15, shots: int = 800) -> float:
    """Oblicza wartość oczekiwaną energii (funkcji celu) na podstawie pomiarów obwodu."""
    qc = build_vqf_ansatz(params, num_qubits=3, reps=1)
    transpiled_qc = transpile(qc, simulator)
    counts = simulator.run(transpiled_qc, shots=shots).result().get_counts()

    total_energy = 0.0
    for bitstring, count in counts.items():
        # Qiskit zwraca bitstring w kolejności [q2, q1, q0]
        y2 = int(bitstring[0])
        y1 = int(bitstring[1])
        x1 = int(bitstring[2])

        cost = classical_factor_cost(x1, y1, y2, N=N)
        total_energy += cost * (count / shots)

    return total_energy


def run_warm_start_vqf(N: int = 15):
    """
    Przeprowadza dwuetapowy proces VQF:
    1. Warm-Start: Szybka optymalizacja wstępna na symulatorze idealnym (0s czasu QPU).
    2. Refinement / Wykonanie końcowe: Pomiary na symulatorze z modelem szumów Odra 5.
    """
    print("=" * 75)
    print(f" VARIATIONAL QUANTUM FACTORING (VQF) DLA N = {N}")
    print("=" * 75)

    sim_ideal = AerSimulator()
    noise_model = create_synthetic_odra5_noise()
    sim_noisy = AerSimulator(noise_model=noise_model)

    # Ansatz z reps=1 potrzebuje 3 + 3 = 6 parametrów
    num_params = 6
    np.random.seed(42)
    initial_params = np.random.uniform(0, 2 * np.pi, num_params)

    print("\n[KROK 1] Wstępny rozruch (Warm-Start) na symulatorze klasycznym (COBYLA)...")
    history_ideal = []

    def callback_ideal(xk):
        e = compute_energy(xk, sim_ideal, N=N, shots=500)
        history_ideal.append(e)

    res_ideal = minimize(
        compute_energy,
        initial_params,
        args=(sim_ideal, N, 500),
        method='COBYLA',
        options={'maxiter': 30},
        callback=callback_ideal
    )

    warm_params = res_ideal.x
    print(f" -> Warm-Start zakończony. Osiągnięta energia minimalna: {res_ideal.fun:.4f}")

    print("\n[KROK 2] Dociągnięcie parametrów i pomiary na procesorze szumowym (Odra 5)...")
    history_noisy = []

    def callback_noisy(xk):
        e = compute_energy(xk, sim_noisy, N=N, shots=800)
        history_noisy.append(e)

    res_noisy = minimize(
        compute_energy,
        warm_params,
        args=(sim_noisy, N, 800),
        method='COBYLA',
        options={'maxiter': 10},
        callback=callback_noisy
    )

    final_params = res_noisy.x

    print("\n[KROK 3] Ostateczny pomiar punktu końcowego (4096 shotów)...")
    qc_final = build_vqf_ansatz(final_params, num_qubits=3, reps=1)

    counts_ideal = sim_ideal.run(transpile(qc_final, sim_ideal), shots=4096).result().get_counts()
    counts_noisy = sim_noisy.run(transpile(qc_final, sim_noisy), shots=4096).result().get_counts()

    # Analiza wyników
    print("\n" + "=" * 50)
    print(" ANALIZA WYNIKÓW ROZKŁADU CZYNINKÓW PIERWSZYCH")
    print("=" * 50)

    for label, counts in [("Symulator Idealny", counts_ideal), ("Fake Odra 5 (Szum)", counts_noisy)]:
        top_bitstring = max(counts, key=counts.get)
        y2, y1, x1 = int(top_bitstring[0]), int(top_bitstring[1]), int(top_bitstring[2])
        p_found = 1 + 2 * x1
        q_found = 1 + 2 * y1 + 4 * y2
        cost_found = classical_factor_cost(x1, y1, y2, N=N)

        print(f"\n--- {label} ---")
        print(f" • Najczęstszy stan bitowy (q2 q1 q0): {top_bitstring} ({counts[top_bitstring]} shotów)")
        print(f" • Odczytane zmienne: x1={x1}, y1={y1}, y2={y2}")
        print(f" • Wyznaczone czynniki: p = {p_found}, q = {q_found}")
        print(f" • Iloczyn p * q = {p_found * q_found} (Wartość funkcji celu: {cost_found})")
        if cost_found == 0:
            print(" -> SUKCES VQF! Wyznaczono poprawne czynniki pierwsze RSA!")

    # Wizualizacja wyników
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_histogram(counts_ideal, ax=ax1, color='darkgreen', title=f"VQF N=15: Idealny Symulator")
    plot_histogram(counts_noisy, ax=ax2, color='darkorange', title=f"VQF N=15: Fake Odra 5 (Z Szumem)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_warm_start_vqf(N=21)