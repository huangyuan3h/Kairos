# Kairos

## install environment

This project is developed and tested with Python 3.8. To create a virtual environment and install the required packages, follow these steps:

1. Create a Conda environment (recommended):

   ```bash
   conda create -n kairos_38 python=3.8
   conda activate kairos_38
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

if install some packages, remember to freeze:

```bash

conda env export > environment.yml

pip freeze > requirements.txt
```








