# NAP (Neural Acquisition Process)

## Setup
Setup a virtualenv/conda/miniconda environment with at least python3.8 and use the requirements.txt to install the dependencies. 
```commandline
# Example with virtualenv
sudo apt-get install python3.8-venv  # for Ubuntu 18.04 LTS
python3.8 -m venv nap_env
. nap_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
 
## Download the datasets
The HPO-Bench data are already present in this .zip to run the small example below.
If you need to download the data for the HPO-B, EDA, MIP and Antigen experiments, you need to download them from this anonymized repository https://osf.io/kbmtu/?view_only=c58ae60d3422469787f83218c34939d8.
```commandline
wget -O NAP_data.zip "https://osf.io/download/u7p8m/?view_only=c58ae60d3422469787f83218c34939d8"
unzip NAP_data.zip
#Train GPs on those datasets with the given scripts
```

## Training
To train NAP on HPO:
```commandline
PYTHONPATH=. python scripts/nap/train_nap_hpo.py
# if it complains about the number of opened files, first run
ulimit -Sn 10000
```

## Testing
Adjust the paths inside scripts/nap/test_nap_hpo.py and run the script.
```commandline
PYTHONPATH=. python scripts/nap/test_nap_hpo.py
```

## License

- https://github.com/metabo-iclr2020/MetaBO is under GNU APGL-3.0 License.

