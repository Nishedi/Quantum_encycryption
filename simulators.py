import json
import re
from typing import Dict, Tuple
import os
from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit.transpiler import CouplingMap
from iqm.qiskit_iqm import IQMProvider

class IQMNoiseBuilder:
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            # self.data = json.load(f)

        self.readout_0to1: Dict[str, float] = {}
        self.readout_1to0: Dict[str, float] = {}
        self.gate1q_fid: Dict[str, float] = {}
        self.gate2q_fid: Dict[str, float] = {}
        self.qubit_names = set()

        self._parse_data()
        self.mapping = self._create_mapping()

    def _parse_data(self):
        for obs in self.data.get('observations', []):
            field = obs.get('dut_field', '')
            val = obs.get('value', 0.0)

            if 'ssro.measure' in field:
                match = re.search(r'\.(QB\d+)\.error_(0_to_1|1_to_0)', field)
                if match:
                    qb = match.group(1)
                    err_type = match.group(2)
                    self.qubit_names.add(qb)
                    if err_type == '0_to_1':
                        self.readout_0to1[qb] = val
                    else:
                        self.readout_1to0[qb] = val

            if 'rb.clifford.xy' in field and 'move' not in field and 'COMPR' not in field:
                match = re.search(r'\.(QB\d+)\.fidelity', field)
                if match:
                    qb = match.group(1)
                    self.qubit_names.add(qb)
                    if qb not in self.gate1q_fid:
                        self.gate1q_fid[qb] = val

            if 'cz' in field and '__COMPR1' in field and 'fidelity' in field:
                match = re.search(r'(QB\d+)__COMPR1', field)
                if match:
                    qb = match.group(1)
                    self.qubit_names.add(qb)
                    self.qubit_names.add('COMPR1')
                    self.gate2q_fid[qb] = val

    def _create_mapping(self) -> Dict[str, int]:
        mapping = {'COMPR1': 0}

        qbs = sorted([q for q in self.qubit_names if q.startswith('QB')],
                     key=lambda x: int(x[2:]))

        for i, qb in enumerate(qbs, start=1):
            mapping[qb] = i

        return mapping

    def build_model(self) -> Tuple[NoiseModel, CouplingMap, Dict[str, int]]:
        nm = NoiseModel()
        coupling_list = []

        avg_01 = sum(self.readout_0to1.values()) / max(1, len(self.readout_0to1))
        avg_10 = sum(self.readout_1to0.values()) / max(1, len(self.readout_1to0))
        avg_1q = sum(self.gate1q_fid.values()) / max(1, len(self.gate1q_fid))
        avg_2q = sum(self.gate2q_fid.values()) / max(1, len(self.gate2q_fid))

        for qb in self.qubit_names:
            if qb == 'COMPR1':
                continue
            idx = self.mapping[qb]

            p01 = self.readout_0to1.get(qb, avg_01)
            p10 = self.readout_1to0.get(qb, avg_10)
            ro_err = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
            nm.add_readout_error(ro_err, [idx])

            fid_1q = self.gate1q_fid.get(qb, avg_1q)
            err_1q = depolarizing_error(1.0 - fid_1q, 1)
            nm.add_quantum_error(err_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'], [idx])

            fid_2q = self.gate2q_fid.get(qb, avg_2q)
            err_2q = depolarizing_error(1.0 - fid_2q, 2)

            coupling_list.append([0, idx])
            coupling_list.append([idx, 0])

            nm.add_quantum_error(err_2q, ['cx', 'cz'], [0, idx])
            nm.add_quantum_error(err_2q, ['cx', 'cz'], [idx, 0])

        coupling_map = CouplingMap(coupling_list)
        return nm, coupling_map, self.mapping


def get_real_iqm_backend(env_file="token.env", verbose: bool = False):
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        print("BRAK TOKENU!")
        raise ValueError("Brak Tokenu")

    server_url = os.getenv("SERVER")

    if not server_url:
        raise ValueError("Brak zmiennej SERVER w środowisku!")
    if verbose:
        print(f"Łączenie z serwerem: {server_url}...")

    provider = IQMProvider(server_url)
    os.environ["IQM_CLIENT_REQUEST_TIMEOUT"] = "100"
    backend = provider.get_backend()
    backend.client._request_timeout = 100
    backend.options.update_options(timeout=100)
    if verbose:
        print(f"Połączono pomyślnie z: {backend.name}")
    return backend

def get_iqm_backend(json_path: str = "calibation_data.json", verbose: bool = False):
    builder = IQMNoiseBuilder(json_path)
    noise_model, coupling_map, mapping_info = builder.build_model()

    sim = AerSimulator(noise_model=noise_model)
    sim.set_options(noise_model=noise_model, coupling_map=coupling_map)
    if verbose:
        print(f"[INFO] Pomyślnie załadowano szum sprzętowy IQM z pliku '{json_path}'.")
        print(f"[INFO] Znaleziono {len(mapping_info) - 1} kubitów peryferyjnych połączonych z centralnym Hubem (COMPR1).")

    print(f"Noise model loaded with {len(mapping_info) - 1} qubits (including COMPR1).")

    return sim


if __name__ == "__main__":
    try:
        print("Inicjalizacja symulatora szumów IQM...")
        backend = get_iqm_backend("calibation_data.json")

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        print("\nPrzed transpilacją (Idealny obwód liniowy):")
        print(qc.draw(fold=-1))

        transpiled_qc = transpile(qc, backend)

        print("\nPo transpilacji na sprzęt IQM (Wymuszenie korzystania z Huba):")
        print(transpiled_qc.draw(fold=-1))
        print("\nSukces! Model szumów oraz CouplingMap działają poprawnie.")

    except FileNotFoundError:
        print("BŁĄD: Nie znaleziono pliku 'calibation_data.json'.")
        print("Skopiuj do folderu plik konfiguracyjny, aby móc załadować fizyczny szum IQM.")

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


def get_odra5_backend_extended(n_qubits: int = 8):
    noise_model = create_synthetic_odra5_noise()
    base_coupling_list = [[0, 2], [1, 2], [2, 3], [2, 4], [2, 0], [2, 1], [3, 2], [4, 2]]

    extended_nm, extended_coupling_list = extend_odra_noise_model(
        noise_model, base_coupling_list, n_qubits=n_qubits, base_n=5
    )

    coupling_map = CouplingMap(extended_coupling_list)
    noisy_sim = AerSimulator(noise_model=extended_nm)
    noisy_sim.set_options(noise_model=extended_nm, coupling_map=coupling_map)

    return noisy_sim

def get_sirius_real_backend():
    from iqm.qiskit_iqm import IQMProvider
    provider = IQMProvider("https://resonance.iqm.tech/", quantum_computer="emerald",
                           token="4hV4IImpLyxaDuj4+E5RhDJp1YIZO3IuEdzef4MiLwQBoBWUpL17YrEkKlbyL3OF")
    backend = provider.get_backend()
    return backend