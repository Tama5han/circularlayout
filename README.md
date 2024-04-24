# circularlayout
This repository contains the Python scripts to draw a circular network. 



## Examples

```python
import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
import random
from circularlayout import CircularLayout

circ = CircularLayout()

node_data = pl.DataFrame({'node': list(range(50))})
edge_data = pl.DataFrame({
    'from': random.choices(node_data['node'].to_list(), k=30),
    'to':   random.choices(node_data['node'].to_list(), k=30)
})

circ.plot(node_data, edge_data, with_labels=True, figsize=(8, 8))
```

You can plota network from networkx.Graph.

```python
G = nx.Graph()

for n in range(50):
    s = random.randint(10, 50)
    G.add_node(n, s=s)

for row in edge_data.iter_rows(named=True):
    w = random.uniform(0.5, 4)
    G.add_edge(row['from'], row['to'], w=w)

circ.plot_from_nx(G, width='w', size='s', with_labels=True, figsize=(8, 8))
```
