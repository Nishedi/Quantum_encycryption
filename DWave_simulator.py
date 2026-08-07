import math
import dimod
from neal import SimulatedAnnealingSampler


class DWave_simulator:

    def __init__(self, N: int, verbose: bool = True):
        self.N = N
        self.verbose = verbose
        self.max_p = int(math.isqrt(N))
        self.max_q = N // 3

        self.num_p_bits = (
            max(1, math.ceil(math.log2((self.max_p - 1) / 2 + 1)))
            if self.max_p >= 3
            else 1
        )
        self.num_q_bits = (
            max(1, math.ceil(math.log2((self.max_q - 1) / 2 + 1)))
            if self.max_q >= 3
            else 1
        )

    def _build_qubo_model(self) -> dimod.BQM:
        p_terms = {(): 1.0}
        for i in range(1, self.num_p_bits + 1):
            p_terms[(f"x_{i}",)] = float(2**i)

        q_terms = {(): 1.0}
        for j in range(1, self.num_q_bits + 1):
            q_terms[(f"y_{j}",)] = float(2**j)

        pq_terms = {}
        for p_var, p_coeff in p_terms.items():
            for q_var, q_coeff in q_terms.items():
                combined_vars = tuple(sorted(p_var + q_var))
                pq_terms[combined_vars] = (
                    pq_terms.get(combined_vars, 0.0) + p_coeff * q_coeff
                )

        diff_terms = pq_terms.copy()
        diff_terms[()] = diff_terms.get((), 0.0) - float(self.N)
        print(f"diff_terms: {diff_terms}")

        hubo_terms = {}
        for term1, coeff1 in diff_terms.items():
            for term2, coeff2 in diff_terms.items():
                combined_vars = tuple(sorted(set(term1 + term2)))
                hubo_terms[combined_vars] = (
                    hubo_terms.get(combined_vars, 0.0) + coeff1 * coeff2
                )

        max_coeff = max(abs(c) for c in hubo_terms.values())
        penalty_strength = max_coeff * 2.0

        bqm = dimod.make_quadratic(
            hubo_terms, strength=penalty_strength, vartype=dimod.BINARY
        )

        max_bqm_val = max(
            max((abs(v) for v in bqm.linear.values()), default=1.0),
            max((abs(v) for v in bqm.quadratic.values()), default=1.0),
        )
        if max_bqm_val > 0:
            bqm.scale(1.0 / max_bqm_val)

        return bqm

    def factor(self, num_reads: int = 500) -> tuple[int | None, int | None]:
        if self.verbose:
            print(f"N = {self.N}")
            print(f"num_p_bits = {self.num_p_bits}, num_q_bits = {self.num_q_bits}")

        bqm = self._build_qubo_model()

        while True:
            pass

        if self.verbose:
            print(
                f"Number of qubo variables: {len(bqm.variables)}"
            )
            print(f"Starting simulated annealing with {num_reads} reads.")

        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=num_reads)

        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy

        if self.verbose:
            print(f"The lowest energy: {best_energy:.4f}")

        p_val = 1 + sum(
            (2**i) * int(best_sample.get(f"x_{i}", 0))
            for i in range(1, self.num_p_bits + 1)
        )
        q_val = 1 + sum(
            (2**j) * int(best_sample.get(f"y_{j}", 0))
            for j in range(1, self.num_q_bits + 1)
        )

        success = (p_val * q_val) == self.N

        if success:
            return p_val, q_val
        return None, None


if __name__ == "__main__":
    ps = [3,5,7,11,13,17,19,23]
    qs = [3,5,7,11,13,17,19,23]
    ps = [3]
    qs = [5]
    for p in ps:
        for q in qs:
            if p >= q:
                continue
            N = p * q
            factorizer = DWave_simulator(N=N, verbose=False)
            p_res, q_res = factorizer.factor(num_reads=500)
            if p_res is not None and q_res is not None:
                if (p_res, q_res) != (p, q) and (p_res, q_res) != (q, p):
                    print(f"Niepoprawny wynik dla N={N}: {p_res}, {q_res}")
                else:
                    print(f"Poprawny wynik dla N={N}: {p_res}, {q_res}")
            else:
                print(f"Nie udało się znaleźć czynników dla N={N}, p={p}, q={q}")
