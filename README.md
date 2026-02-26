Generate topological orderings by running 
``` text, edge_list = generate_random_dag(n=n, k=k, seed=seed)```
```text```: str looks like this
6->0
5->3
1->12
16->12
16->3
17->2
6->7
7->13
6->11
1->14
8->3
8->0
10->11
6->15
13->9
5->10
16->8
2->7
16->9
12->13

edge_list looks like this:
[(6, 0), (5, 3), (1, 12), (16, 12), (16, 3), (17, 2), (6, 7), (7, 13), (6, 11), (1, 14), (8, 3), (8, 0), (10, 11), (6, 15), (13, 9), (5, 10), (16, 8), (2, 7), (16, 9), (12, 13)]

Probably better to feed edge list into the LLM tokenizer after making it a string

to get a valid ground truth topological ordering (there may be many):
ordering = generate_topological_sort(edge_list, n=n)

to check if a toplogical ordering is a valid topological stor:
is_topological_ordering(ordering=ordering, dag=edge_list, n=n) # (where n gives the number of nodes in the graph)


you'll need to download the qwen tokenizer and model from hugging face 

2. hugginface-cli login
1. huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir ./models/qwen2.5-0.5b-instruct \
  --local-dir-use-symlinks False