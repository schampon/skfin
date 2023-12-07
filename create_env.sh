conda create python=3.9 --name  skfin -c https://conda.anaconda.org/conda-forge/ -y
conda activate skfin

pip install -r requirements.txt
pip install -e . 
python -m ipykernel install --user --name skfin --display-name "Python (skfin)"
