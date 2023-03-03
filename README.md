# Fairness-in-Learning-of-LDS

This the source code for the paper *Fairness in Forecasting of Observations of Linear Dynamical Systems*:

```
@misc{https://doi.org/10.48550/arxiv.2209.05274,
  doi = {10.48550/ARXIV.2209.05274},
  url = {https://arxiv.org/abs/2209.05274},
  author = {Zhou, Quan and Marecek, Jakub and Shorten, Robert N.},
  title = {Fairness in Forecasting of Observations of Linear Dynamical Systems},
  publisher = {arXiv},
  year = {2022},
}
```

## Dependencies

1. Mosek/9.2 https://www.mosek.com/downloads/list/9/

2. Python scripts:

- Python/3.9.6

- inputlds https://raw.githubusercontent.com/jmarecek/OnlineLDS/master/inputlds.py

- ncpol2sdpa 1.12.2 https://ncpol2sdpa.readthedocs.io/en/stable/index.html

- AIF360 (for post-processing) https://github.com/Trusted-AI/AIF360

3. Julia scripts:

- Julia/1.8.5

- TSSOS https://github.com/wangjie212/TSSOS

4. The COMPAS dataset: 

- It was downloaded from https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis
- It is named as ``data/compas-scores-two-years.csv''.

## The structure of scripts:

- F1.py <font size='1'> dd
- F2.py
- F3_ncpol2sdpa.py F3_tssos.py F3_tssos_compas.py F3_sparsity.py
- F4.py
- PostProcess_1.py PostProcess_2.py PostProcess_aif360.ipynb
- PostProcess_plot.ipynb
- functions.py
- fairncpop.batch 
