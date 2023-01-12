
<h1 align="center">
  <br>
  
  ![MPFC](README/pccc.gif)

</h1>

An algorithm for clustering with hard and soft must-link and cannot-link constraints. A detailed description of the algorithm can be found in https://arxiv.org/abs/2212.14437.

## Dependencies

PCCC depends on:
* pandas==1.5.2
* scikit-learn==1.0.2
* numpy==1.23.5
* networkx==2.7.1
* scipy==1.9.3
* gurobipy==10.0.0

Gurobi is a commercial mathematical programming solver. Free academic licenses are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/). 

## Installation

1) Clone this repository
2) Download and install Gurobi (https://www.gurobi.com/downloads/)
3) Install the other required packages

## Usage

The main.py file contains code that applies the PCCC algorithm to an illustrative example.

```python
labels = pccc(X, n_clusters,
              ml=hard_mustlink_constraints,
              cl=hard_cannotlink_constraints,
              sml=soft_mustlink_constraints,
              scl=soft_cannotlink_constraints,
              sml_weights=confidence_levels_of_soft_mustlink_constraints,
              scl_weights=confidence_levels_of_soft_cannotlink_constraints,
              random_state=24)
```

## Reference

Please cite the following paper if you use this algorithm.

**Baumann, P. and Hochbaum D. S.** (2023): PCCC: The Pairwise-Confidence-Constraints-Clustering Algorithm. https://arxiv.org/abs/2212.14437

Bibtex:
```
@article{baumann2023pccc,
	author={Baumann, Philipp and Hochbaum, Dorit S.},
	booktitle={},
	title = {PCCC: the pairwise-confidence-constraints-clustering algorithm},
	year={2023},
	url = {https://arxiv.org/abs/2212.14437},
	doi = {10.48550/ARXIV.2212.14437},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


