# TimeMatch
Official source code of [TimeMatch: Unsupervised Cross-region Adaptation by Temporal Shift Estimation](https://arxiv.org/abs/2111.02682) by Joachim Nyborg, [Charlotte Pelletier](https://sites.google.com/site/charpelletier/), [Sébastien Lefèvre](http://people.irisa.fr/Sebastien.Lefevre/), and Ira Assent.

## Requirements
### Python requirements
- Python 3.9.4, PyTorch 1.8.1, and more in `environment.yml`.

### TimeMatch dataset download
The dataset can be freely downloaded from [Zenodo](https://doi.org/10.5281/zenodo.5636422).
The extracted size is about 78 GB.

### Pre-trained models and full results
Pre-trained models and results can also be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.5636422)


## Usage
Setup conda environment and activate
```
conda env create -f environment.yml
conda activate timematch
```

Download dataset and extract to `/media/data/timematch_data` (or set `--data_root` to its path for `train.py`).

Pre-trained models should be extracted to `timematch/outputs`.

Example: train model on the source domain
```
python train.py -e pseltae_32VNH --source denmark/32VNH/2017 --target denmark/32VNH/2017
```

Train TimeMatch with pre-trained model
```
python train.py -e timematch_32VNH_to_30TXT --source denmark/32VNH/2017 --target france/30TXT/2017 timematch --weights outputs/pseltae_32VNH
```

All training scripts can be found in the `scripts` directory.


## Reference
If you find TimeMatch and the code useful, please consider citing our paper using the following BibTeX entry.
```
@article{nyborg2021timematch,
  title={TimeMatch: Unsupervised Cross-Region Adaptation by Temporal Shift Estimation},   
  author={Joachim Nyborg and Charlotte Pelletier and Sébastien Lefèvre and Ira Assent},
  year={2021},
  journal={arXiv preprint arXiv:2111.02682}
}
```

## Credits
- The implementation of PSE+LTAE is based on [the official implementation](https://github.com/VSainteuf/lightweight-temporal-attention-pytorch)
- The implementation of competitors MMD, DANN, and CDAN is based on [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library),
and JUMBOT on [its official implementation](https://github.com/kilianFatras/JUMBOT).
- The annotations used in the TimeMatch dataset from the LPIS produced by the mapping agencies [IGN](https://www.data.gouv.fr/en/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire) (France), 
[NaturErhverstyrelsen](https://kortdata.fvm.dk) (Denmark), and the [AMA](https://www.data.gv.at/katalog/dataset/d3b0cdeb-5727-46dd-8de4-a76f1898fd9b) (Austria).
- The Sentinel-2 imagery were accessed from the [AWS bucket managed by Sinergise](https://registry.opendata.aws/sentinel-2/)


