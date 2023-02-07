import json
import math
import time
from random import shuffle, seed
from networkx import Graph, fast_gnp_random_graph
from srs import StochasticSort as srs

input_sz = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
c_value = [1, 5, 10]
sample_sz = 10

samples = [(c_v, sz, k) for c_v in c_value for sz in input_sz for k in range(sample_sz)]
results = {c_v: {sz: [] for sz in input_sz} for c_v in c_value}
shuffle(samples)


def quick_gen(n, p):
    weights = [i for i in range(n)]
    shuffle(weights)
    sol = [weights.index(i) for i in range(n)]
    graph = Graph()
    for indx, value in enumerate(weights):
        graph.add_node(indx, weight=value)
    for i in range(len(sol) - 1):
        graph.add_edge(sol[i], sol[i + 1])
    ograph = fast_gnp_random_graph(n,p)
    for u, v in ograph.edges:
        graph.add_edge(u,v)
    return graph, sol


if __name__ == "__main__":
    seed(5)  # current data not from this seed.
    trial = 1
    for c_v, sz, _ in samples:
        print(f"Trial {trial}")
        trial += 1
        print(c_v, sz, len(results[c_v][sz]))
        pv = math.log(sz)/sz
        test_input, test_sol = quick_gen(sz, pv)
        print(test_input.size(), pv)
        # running trial
        st = time.perf_counter_ns()
        test = srs(test_input, pv, c_v)
        output = test.sort()
        et = time.perf_counter_ns()
        assert test_sol == output
        print(et-st, test.queries)
        results[c_v][sz].append((test.queries,et-st))
        with open("results.json", 'w') as data_out:
            json.dump(results, data_out, indent=2)
