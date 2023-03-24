# The Vesuvius Challenge
*This repository contains codebase for the competition as well as setup for experiments version control for "UCU dropouts" team*


# Setup
## Local setup
### Prerequisites
- Python 3 (3.10 preferable)
- `pip`, `venv` modules available
On the fresh systems these things might be absent


### Environment setup
First of all you will need to create python virtual environment and install necessary packages:
```
python3 -m venv venv
source venv/bin/activate
pip install -e . --no-deps
pip install -r requirements.txt
```

To train locally, on your own host machine, it's required to download kaggle competition dataset first.
It's suggested to store it in the `dataset` folder.
To use `kaggle cli`, you have to setup your credentials according to this [instruction](https://github.com/Kaggle/kaggle-api#api-credentials).
To download dataset use the following command:
```
kaggle competitions download -c vesuvius-challenge-ink-detection --force --path dataset
unzip dataset/vesuvius-challenge-ink-detection.zip -d dataset
```
*Note: Takes 15 mins to download with speed 25 MB/s (UCU WIFI) and 27 mins to extract archives on HDD.*


## Colab setup
**Work in progress**
The idea is to have a notebook with possibility to specify params and clone this repo and execute trainings with possibly some local changes. The credentials
Good hint is to download the dataset to Google Drive to decrease time needed to initialize the dataset.
The credentials might be stored as Google Drive file.


## Kaggle setup
**Work in progress**
The idea is to have a notebook with possibility to specify params, git clone this repo and execute trainings with possibly some local changes. The credentials to git clone should be shared through secret variables on kaggle.


# Training
**TBD**

# Submission

**Work in progress**

The main idea behind it is to have Jupyter notebook `.ipynb` synchronized with this repo and kaggle, shared across team, so everyone could update models and run evaluation on kaggle hidden dataset.

# Experiments control

All the experiments should be tracked in some system, the proposed one is `Weights&Biases`.
According to this system, all the artifacts might be stored directly on their cloud servers.