# ./tests/test.py
import re
import pytest
import sys

# IMPORTANT:
# Change this import to match wherever you put the functions under test.
# Example: from src.toposort_data import generate_random_dag, generate_topological_sort
sys.path.append("..")
from utils import generate_random_dag, generate_topological_sort  # <-- EDIT THIS


def _parse_graph_text(graph_text: str):
    """
    Parse lines like:
        1->2
        5->17
    into [(1,2), (5,17)].

    Accepts empty string => [].
    """
    graph_text = graph_text.strip()
    if not graph_text:
        return []

    edges = []
    for line in graph_text.splitlines():
        line = line.strip()
        m = re.fullmatch(r"(\d+)->(\d+)", line)
        assert m is not None, f"Bad line format: {line!r}"
        u = int(m.group(1))
        v = int(m.group(2))
        edges.append((u, v))
    return edges


def _is_topological_ordering(ordering, edges, n):
    """
    True iff ordering is a permutation of 0..n-1 and respects all edges u->v.
    """
    if len(ordering) != n:
        return False
    if set(ordering) != set(range(n)):
        return False
    pos = {node: i for i, node in enumerate(ordering)}
    return all(pos[u] < pos[v] for (u, v) in edges)


# -----------------------------
# Tests for generate_random_dag
# -----------------------------

def test_generate_random_dag_n0_k0():
    text, edges = generate_random_dag(n=0, k=0, seed=123)
    assert text == ""
    assert edges == []


def test_generate_random_dag_raises_if_too_many_edges():
    # max edges in DAG with n nodes is n*(n-1)/2
    n = 5
    max_edges = n * (n - 1) // 2
    with pytest.raises(ValueError):
        _ = generate_random_dag(n=n, k=max_edges + 1, seed=0)


def test_generate_random_dag_text_matches_edge_list_exactly():
    text, edges = generate_random_dag(n=10, k=12, seed=7)

    parsed = _parse_graph_text(text)
    # Exact match including ordering of edges; generator uses the edge_list order for text.
    assert parsed == edges


def test_generate_random_dag_edges_are_in_range_and_unique():
    n = 20
    k = 50
    _, edges = generate_random_dag(n=n, k=k, seed=42)

    assert len(edges) == k
    assert len(set(edges)) == k  # should be unique because of random.sample

    for u, v in edges:
        assert 0 <= u < n
        assert 0 <= v < n
        assert u != v  # no self edges in this generator


def test_generate_random_dag_is_deterministic_with_seed():
    text1, edges1 = generate_random_dag(n=15, k=30, seed=999)
    text2, edges2 = generate_random_dag(n=15, k=30, seed=999)

    assert text1 == text2
    assert edges1 == edges2


def test_generate_random_dag_is_probabilistic_without_seed():
    # Not guaranteed, but extremely likely for these sizes.
    text1, edges1 = generate_random_dag(n=25, k=60, seed=None)
    text2, edges2 = generate_random_dag(n=25, k=60, seed=None)

    assert (text1 != text2) or (edges1 != edges2)


# ----------------------------------
# Tests for generate_topological_sort
# ----------------------------------

def test_generate_topological_sort_empty_graph_n0():
    ordering = generate_topological_sort(dag=[], n=0)
    assert ordering == []  # vacuously fine


def test_generate_topological_sort_single_node_no_edges():
    ordering = generate_topological_sort(dag=[], n=1)
    assert ordering == [0]


def test_generate_topological_sort_disconnected_graph():
    # n=6, only one edge; others disconnected
    edges = [(2, 5)]
    ordering = generate_topological_sort(dag=edges, n=6)
    assert _is_topological_ordering(ordering, edges, n=6)


def test_generate_topological_sort_simple_chain():
    edges = [(0, 1), (1, 2), (2, 3)]
    ordering = generate_topological_sort(dag=edges, n=4)
    assert ordering == [0, 1, 2, 3]  # with this algorithm, this is forced
    assert _is_topological_ordering(ordering, edges, n=4)


def test_generate_topological_sort_cycle_returns_empty():
    edges = [(0, 1), (1, 2), (2, 0)]
    ordering = generate_topological_sort(dag=edges, n=3)
    assert ordering == []


def test_generate_topological_sort_self_loop_returns_empty():
    edges = [(0, 0)]
    ordering = generate_topological_sort(dag=edges, n=1)
    assert ordering == []


def test_generate_topological_sort_valid_on_random_dags_many_trials():
    # Property-style test: generated DAG -> returned ordering should be a valid topo ordering
    for seed in range(25):
        n = 30
        k = 80
        _, edges = generate_random_dag(n=n, k=k, seed=seed)
        ordering = generate_topological_sort(dag=edges, n=n)
        assert ordering != []
        assert _is_topological_ordering(ordering, edges, n=n)


def test_generate_topological_sort_raises_on_out_of_range_edges():
    # Your function does not explicitly validate edge endpoints.
    # These inputs should raise IndexError (inbound_count[child]) or KeyError (adj[parent]).
    with pytest.raises((IndexError, KeyError)):
        _ = generate_topological_sort(dag=[(0, 3)], n=3)  # child out of range

    with pytest.raises((IndexError, KeyError)):
        _ = generate_topological_sort(dag=[(3, 0)], n=3)  # parent out of range