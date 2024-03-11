# Hodge-Compositional Edge Gaussian Processes

This is the code for Hodge-Compositional (HC) edge Gaussian processes (GPs). It contains the three experiments, together with the corresponding datasets, as well as the kernel classes. 


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



## License
The project is listed under the [MIT](https://choosealicense.com/licenses/mit/) license.


