
<h1 align="center">
    PCCC: An Algorithm for Clustering with Confidence-Based Must-Link
and Cannot-Link Constraints
  <br>
  
  ![MPFC](README/cover.png)

[![License](https://img.shields.io/badge/License-MIT_License-blue)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-INFORMS_Journal_on_Computing-green)](https://doi.org/10.1287/ijoc.2023.0419)

</h1>

The PCCC algorithm is a clustering method that can deal with both hard and soft must-link and cannot-link constraints. Individual confidence values in (0, 1] can be provided 
for all soft constraints. The higher the confidence value, the harder the algorithm tries to satisfy it. A detailed description of the algorithm can be found in our paper https://doi.org/10.1287/ijoc.2023.0419. 

## Installation

1) Clone this repository
2) Install Gurobi (https://www.gurobi.com/). Gurobi is a commercial mathematical programming solver. Free academic licenses are available.
3) Create a virtual environment

```
python -m venv venv
```

4) Activate the virtual environment and install the required Python packages using the following command: 

```
pip install -r requirements.txt
```


## Usage

The run_illustrative_example.py file contains code that applies the PCCC algorithm to an illustrative example.

```python
output = pccc(X, n_clusters,
              ml=hard_must_link_constraints,
              cl=hard_cannot_link_constraints,
              sml=soft_must_link_constraints,
              scl=soft_cannot_link_constraints,
              sml_weights=confidence_levels_of_soft_must_link_constraints,
              scl_weights=confidence_levels_of_soft_cannot_link_constraints,
              cluster_repositioning='violations_inertia',  # ='none'
              dynamic_n_neighbors='n_violations_neighbors.500.10.after_repositioning',  # ='none'
              random_state=4)
```

The run_instance.py file demonstrates how to load an instance from a file and apply the PCCC algorithm to it.

## Reference

Please cite the following paper if you use this algorithm.

**Baumann, P. and Hochbaum D. S.** (2024): An algorithm for clustering with confidence-based must-Link and cannot-link constraints. INFORMS Journal on Computing. https://doi.org/10.1287/ijoc.2023.0419.

Bibtex:
```
@article{baumann2024pccc,
	author={Baumann, Philipp and Hochbaum, Dorit S.},
	journal={INFORMS Journal on Computing},
	title = {An algorithm for clustering with confidence-based must-Link and cannot-link constraints},
	year={2024},
	url = {https://doi.org/10.1287/ijoc.2023.0419},
	doi = {10.1287/ijoc.2023.0419},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


