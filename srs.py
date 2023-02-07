from functools import reduce
from typing import Tuple, List, Set, Dict, Optional, Any

from networkx import Graph
from math import log, ceil
from random import uniform, seed, shuffle


def flip_fact(threshold):
    def flipper():
        value = uniform(0, 1)
        if value < threshold:
            return True
        else:
            return False

    return flipper


class StochasticSort:
    def __init__(self, graph: Graph, p: float, c: int = 10):
        self.queries = 0
        self.p: float = p
        self.c: int = c
        self.q: int = max(1, ceil(log(len(graph) * p, 2)))  # gets the total number of levels
        self.alpha: float = self.solve_alpha(p, self.q)
        self.graph = graph
        self.target_sizes = {i: 2 ** i / self.p for i in range(1, self.q + 1)}  # target sizes for levels
        self.edge_set: Dict[int, Set[Tuple[int, int]]] = self.generate_edge_set()

    @staticmethod
    def solve_alpha(p, q):
        # find the alpha value using binary search with 6 sig figs.
        def eval_alpha(a):
            output = 1.0
            for i in range(q):
                output *= (1 - p * a / 2.0 ** (i + 1))
            return output - 1 + p

        c_a = 1.5
        err = eval_alpha(c_a)
        upper = 2
        lower = 1
        while abs(err) > 10 ** (-6):
            if err > 0:
                lower = c_a
                c_a = (upper + c_a) / 2.0
                err = eval_alpha(c_a)
            else:
                upper = c_a
                c_a = (c_a + lower) / 2.0
                err = eval_alpha(c_a)
        return c_a

    def sort(self):
        output = []
        levels, blocks = self.build_levels(None, None)
        output.append(
            self.find_first()
        )
        self.update_levels(levels, blocks, output[-1])
        while len(output) != len(self.graph):
            next_element = self.get_next(output, levels)
            output.append(next_element)
            self.update_levels(levels, blocks, next_element)
            l = len(output)  # this is the l in x_l from the paper. Denotes current iteration
            last_level_to_rebuild = 0
            # find top level that needs to be rebuilt.
            for level in range(self.q, 0, -1):
                if l % ceil(self.target_sizes[level] / 32) == 0:
                    last_level_to_rebuild = level
                    break
            # modify current levels with rebuild
            levels, blocks = self.build_levels(levels, blocks, last_level_to_rebuild)
        return output

    def generate_edge_set(self) -> Dict[int, Set[Tuple[int, int]]]:
        # produces a set Bernoulli variables with specified probability
        generators = [flip_fact(self.alpha * self.p / (2 ** (i + 1))) for i in range(self.q)]
        edge_set = {i + 1: set() for i in range(self.q)}
        for u, v in self.graph.edges:
            sample = [gen() for gen in generators]
            while not reduce(lambda x, y: x or y, sample):  # interpreter interpreting "any" as a type so reduce
                # sample = List[Tuple[int, int]]
                sample = [gen() for gen in generators]
            for indx, b_value in enumerate(sample):
                if b_value:
                    edge_set[indx + 1].add((u, v))
        return edge_set

    def build_levels(self, levels: Optional[Dict[int, Set[int]]],
                     blocks: Optional[Dict[int, Dict[int, Optional[int]]]],
                     n: Optional[int] = None) \
            -> Tuple[Dict[int, Set[int]], Dict[int, Dict[int, int]]]:
        if n is None:  # initializes the blocks
            levels = {self.q + 1 + i: {vertex for vertex in self.graph} for i in range(self.c)}
            blocks = {indx: {vtx: None for vtx in level} for indx, level in levels.items()}
            for i in range(self.q, 0, -1):
                self.construct_level(levels, blocks, i)
            return levels, blocks
        else:
            for i in range(n, 0, -1):
                self.construct_level(levels, blocks, i)
            return levels, blocks

    def construct_level(self, levels: Dict[int, Set[int]],
                        blocks: Dict[int, Dict[int, Optional[int]]],
                        i: int) -> None:
        next_level: Set = set()
        new_block: Dict = dict()
        for v in levels[i + 1]:
            for u in levels[i + self.c]:
                if (u, v) in self.edge_set[i] or (v, u) in self.edge_set[i]:
                    if self.query(u, v):
                        new_block[v] = u
                        break
            else:
                next_level.add(v)
                new_block[v] = None
        levels[i] = next_level
        blocks[i] = new_block

    def query(self, u: int, v: int) -> bool:
        self.queries += 1
        return self.graph.nodes[u]["weight"] < self.graph.nodes[v]["weight"]

    def find_first(self) -> Any:
        init_edge = next(iter(self.graph.edges))  # pull random edge from the graph
        smallest = init_edge[0] if self.query(*init_edge) else init_edge[1]
        checked = set(init_edge)
        while True:
            # check if all neighbors of the smallest node has already been checked
            if {vtx for vtx in self.graph.neighbors(smallest)}.issubset(checked):
                break
            # check if any unchecked neighbor is smaller
            for u in self.graph.neighbors(smallest):
                if u not in checked:
                    if self.query(u, smallest):
                        smallest = u
                        checked.add(u)
                        break
                    else:
                        checked.add(u)
        return smallest

    def check_for_block(self, vtx: Any, indx: int,
                        levels: Dict[int, Set[int]],
                        blocks: Dict[int, Dict[int, Optional[int]]]) -> bool:
        for u in levels[indx + self.c]:
            if (u, vtx) in self.edge_set[indx] or (vtx, u) in self.edge_set[indx]:
                if self.query(u, vtx):
                    blocks[indx][vtx] = u
                    return True
        blocks[indx][vtx] = None
        return False

    def get_next(self, current_elements: List[int], levels: Dict[int, Set[int]]) -> int:
        candidates = set(filter(lambda u: self.graph.has_edge(current_elements[-1], u), levels[1]))
        while len(candidates) != 1:
            candidate = next(iter(candidates))
            for i in range(1, self.q + self.c + 1):
                candidate_dead = False
                for vtx in levels[i]:
                    if self.graph.has_edge(vtx, candidate):
                        if self.query(vtx, candidate):
                            candidates.remove(candidate)
                            candidate_dead = True
                            break
                if candidate_dead:
                    break
            else:
                return candidate
        return candidates.pop()

    def update_levels(self, levels: Dict[int, Set[int]],
                      blocks: Dict[int, Dict[int, Optional[int]]],
                      next_element: int) -> None:
        for i in range(self.q + self.c, 0, -1):
            if next_element in levels[i]:
                levels[i].remove(next_element)
                del blocks[i][next_element]
        for i in range(self.q + self.c, 0, -1):
            for vtx, block in blocks[i].items():
                if block == next_element:
                    current_level = i
                    while current_level >= 1:
                        if self.check_for_block(vtx, current_level, levels, blocks):
                            break
                        levels[current_level].add(vtx)
                        current_level -= 1


if __name__ == "__main__":
    seed(12)
    sz = 100
    _p = .5
    weights = [i for i in range(sz)]
    shuffle(weights)
    print(weights)
    sol = [weights.index(i) for i in range(sz)]
    print([weights.index(i) for i in range(sz)])
    test = Graph()
    for _indx, _value in enumerate(weights):
        test.add_node(_indx, weight=_value)
    for i in range(len(sol) - 1):
        test.add_edge(sol[i], sol[i + 1])
    _flipper = flip_fact(_p)
    for _indx, _value in enumerate(weights):
        for j in range(_indx):
            if _flipper() and not test.add_edge(j, _indx):
                test.add_edge(j, _indx)
    print(test.size())
    test_value = StochasticSort(test, _p, c=4)
    _level, _blocks = test_value.build_levels(None, None)
    for key in sorted(_level.keys()):
        print(f"{_level[key]}, {_blocks.get(key)}")
    print(test_value.sort())
    print(test_value.sort() == sol)
    # should be 1,2,0,5,3,4
