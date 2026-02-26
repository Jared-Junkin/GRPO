import random
from typing import List, Tuple
from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader



class GRPODataset(Dataset):
    def __init__(self, ds: list[str])->None:
        # ds: a list of graphs where each graph is stored as a edge list (str): "[(6, 0), (5, 3), (1, 12), (16, 12), (16, 3), (17, 2), (6, 7), (7, 13), (6, 11), (1, 14), (8, 3), (8, 0), (10, 11), (6, 15), (13, 9), (5, 10), (16, 8), (2, 7), (16, 9), (12, 13)]"
        super().__init__()
        self.ds = ds
    def __len__(self)->int:
        return len(self.ds)
    def __getitem__(self, index: int)->str:
        return self.ds[index]

def generate_dataset(num_graphs: int, writefile: str = None, n:int = 20, k: int=20)->list[str]:
    '''
    num_graphs: number of graphs to generate
    writefile: full or relative path to text file to write graphs to (./data/train_ds.txt)
    n: number of nodes in each graph
    k: number of edges in each graph
    '''
    ds = []
    for seed in range(num_graphs):
        text, edge_list = generate_random_dag(n=n, k=k, seed=seed) # not using text right now
        ds.append(str(edge_list))

    if writefile is not None:
        with open(writefile, 'w') as f:
            f.write("\n".join(ds))
    return ds

def load_dataset(writefile: str)->list[str]:
    with open(writefile, 'r') as f:
        ds = [line.strip() for line in f]
    return ds
def generate_random_dag(n: int, k: int, seed: int | None = None) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Generate a random DAG with n nodes and k edges.

    Returns:
        graph_text: str formatted like:
            "1->2\n2->3\n..."
        edge_list: List[(from, to)]
    """
    if seed is not None:
        random.seed(seed)

    if k > n * (n - 1) // 2:
        raise ValueError("Too many edges for a DAG with n nodes.")

    # Nodes labeled 0...n-1
    nodes = list(range(0,n))

    # Sample a random topological ordering
    topo_order = nodes[:]
    random.shuffle(topo_order)

    # All valid forward edges under this ordering
    possible_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            u = topo_order[i]
            v = topo_order[j]
            possible_edges.append((u, v))

    # Sample k edges without replacement
    edge_list = random.sample(possible_edges, k)

    # Format as text
    graph_text = "\n".join(f"{u}->{v}" for u, v in edge_list)

    return graph_text, edge_list

# now let's write code to do toplogical sort on this DAG
def generate_topological_sort(dag: list[tuple], n: int)->list:
    # generate adjacenciy list (for each node, list out the child nodes)
    adj = {i: [] for i in range(n)}
    # generate reverse adjacency list: the total in degree of each node
    inbound_count = [0]*n
    for parent, child in dag:
        adj[parent].append(child)
        inbound_count[child] += 1

    # add all nodes that have in degree = 0 to a FIFO queue
    queue = deque([i for i in range(len(inbound_count)) if inbound_count[i]==0])

    # list = []
    ordering = []
    # while queue is not empty
    while queue:
        # node = pop queue
        node = queue.pop()

        # add node to list
        ordering.append(node)
        # for child in adj[node]
        for child in adj[node]:
            inbound_count[child] -= 1
            if inbound_count[child] == 0:
                queue.append(child)
    
    if len(ordering)==n:
        return ordering
    else:
        return [] # no topological sort possible
        
# how do I determine if something is in a topological sort
def is_topological_ordering(*, ordering: list[int], dag: list[tuple[int, int]], n: int) -> bool:
    # ordering must contain every node exactly once
    if len(ordering) != n:
        return False
    if set(ordering) != set(range(n)):
        return False

    # build a fast lookup: for each node, what index does it appear at in the ordering?
    position = {node: i for i, node in enumerate(ordering)}

    # definition: ordering is topological iff for every edge (u -> v),
    # u appears before v in the ordering
    for parent, child in dag:
        if position[parent] >= position[child]:
            return False

    return True

if __name__ == "__main__":
    # generate 1000 graphs
    n=20
    k=20
    seed=16
    text, edge_list = generate_random_dag(n=n, k=k, seed=seed)
    print(text)
    print(edge_list)
    # for seed in range(1000):
    #     text, edge_list = generate_random_dag(n=n, k=k, seed=seed)
    #     ordering = generate_topological_sort(edge_list, n=n)
    #     print(f"start of edge list {seed}: {edge_list[:10]}, is topological ordering? {is_topological_ordering(ordering=ordering, dag=edge_list, n=n)}")

