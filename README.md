# Kairos

## install environment

This project is developed and tested with Python 3.12. To create a virtual environment and install the required packages, follow these steps:

1. Create a Conda environment (recommended):

   ```bash
   conda create -n stock_pred_312 python=3.12
   conda activate stock_pred_312
   ```
   
ps: deactivate environment:

```bash
conda deactivate
```
   
2. Install dependencies:

```bash
conda env create -f environment.yml

pip install -r requirements.txt


```



----- in developing



```bash
conda create -n kairos_312 python=3.12

conda activate kairos_312

conda deactivate


conda install akshare pandas 


conda env remove <env name>

conda env export > environment.yml

```


conda create -n stock_prediction python=3.8
