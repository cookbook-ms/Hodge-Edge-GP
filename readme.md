# Hodge-Compositional Edge Gaussian Processes

This is the code for the paper [Hodge-Compositional (HC) edge Gaussian processes (GPs)](https://arxiv.org/abs/2310.19450) in AISTATS 2024. It contains the three experiments, together with the corresponding datasets, as well as the kernel classes. 


## Installation

create an environment and install the needed packages

```bash
conda env create -f hc-gp-environment.yml
```

## Usage
We provide Jupyter Notebook files for the three experiments. 

In [GPRegression_Forex](GPRegression_Forex.ipynb), we provide the tutorial of using edge Gaussian processes to interpolate a foreign currency exchange (Forex) market. 
The dataset is [forex_2018.pkl](data/forex/forex_2018.pkl). 
The kernels are written using [GPyTorch](https://docs.gpytorch.ai/en/stable/) in [edge_kernel_forex](kernels/edge_kernel_forex.py). 
One can select the following kernels implmented:
- HC Maten or diffusion kernel 
- non-HC Matern or diffusion kernel
- Euclidean Matern or rbf kernel 
- Line-graph Matern or diffusion kernel 

Likewise, in [GPRegression_OceanFlow](GPRegression_OceanFlow.ipynb) and [GPRegression_WSN](GPRegression_WSN.ipynb), we provide the tutorials for interpolating the ocean edge flows and a water supply network. 

For example, to run the Forex experiment with the HC edge Matern kernel with 20\% training ratio, one can bash the following 
```bash
run -i GPRegression_Forex.ipynb --seed 5 --kernel_name matern --train_ratio 0.2
```


## Citation 
```
@InProceedings{pmlr-v238-yang24e,
  title = 	 {Hodge-Compositional Edge {G}aussian Processes},
  author =       {Yang, Maosheng and Borovitskiy, Viacheslav and Isufi, Elvin},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {3754--3762},
  year = 	 {2024},
  editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v238/yang24e/yang24e.pdf},
  url = 	 {https://proceedings.mlr.press/v238/yang24e.html}
}
```



## License
The project is listed under the [MIT](https://choosealicense.com/licenses/mit/) license.


