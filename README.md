# Emotion Embeddings

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7405327.svg)](https://doi.org/10.5281/zenodo.7405327)


This repository holds an extension of the codebase behind the EMNLP 2021 
[paper](http://doi.org/10.18653/v1/2021.emnlp-main.728) 
*Towards Label-Agnostic Emotion Embeddings* to facial emotion recognition. It thus generalizes our 
emotion-embedding-approach from language _only_ to language _and_ vision.

By Sven Buechel and Udo Hahn, Jena University Language and Information Engineering (JULIE) Lab: https://julielab.de. With special thanks to 
Luise Modersohn for participating in an earlier version of this codebase.

## Installation
This codebase was developed on and tested for Debian 9.

`cd` into the project root folder. Set up your conda environment: 
```
conda create --name "emocoder" python=3.7 pip
conda activate emocoder
pip install -r requirements.txt
```

Add the project root to your PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$(pwd)`.

Copy and rename the file `emocoder/experiments/config_template.json` to `config.json`.

You are now set-up to use our codebase. However, to re-run our experiments, you will need to download the respective 
datasets (see below).


## Datasets

There are four types of datasets necessary to replicate all experiments, text datasets, word datasets, image datasets,  
and word embeddings.

### Word Embeddings

Download the following files (1–2GB each), unzip them and place them under `emocoder/data/vectors`.

- [English](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip)
- [Spanish](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz)
- [German](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz)
- [Polish](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.vec.gz)
- [Turkish](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.vec.gz)


### Word Datasets

- en1. Either request the **1999**-version of the Affective Norms for English Words (ANEW) from the 
[Center for the Study of Emotion and Attention](https://csea.phhp.ufl.edu/Media.html#bottommedia) at the University 
of Florida, or copy-paste/parse the data from the Techreport  *Bradley, M. M., & Lang, P. J. (1999). Affective Norms 
for English Words (Anew): Stimuli, Instruction Manual and Affective Ratings (C–1). The Center for Research in 
Psychophysiology, University of Florida.* Format the data as an tsv file with column headers `word`, `valence`, 
`arousal`, `dominance` and save it under `emocoder/data/datasets/ANEW1999.csv`.
- en2. Get the file `Stevenson(2007)-ANEW_emotional_categories.xls` from [Stevenson et al. (2007)](https://doi.org/10.3758/BF03192999)
 and save it as `emocoder/data/datasets/stevenson2007.xls`.
- es1. Get the file `13428_2015_700_MOESM1_ESM.csv` from [Stadthagen-Gonzalez et al. (2017)](https://doi.org/10.3758/BF03192999) 
and save it as `emocoder/data/datasets/Stadthagen_VA.csv`.
- es2. Get the file `13428_2017_962_MOESM1_ESM.csv` from [Stadthagen-Gonzalez et al. (2018)](https://doi.org/10.3758/s13428-017-0962-y) 
and save it as `emocoder/data/datasets/Stadthagen_BE.csv`.
- de1. Get the file `BAWL-R.xls` from [Vo et al. (2009)](https://doi.org/10.3758/BRM.41.2.534) which is currently available 
[here](https://www.ewi-psy.fu-berlin.de/einrichtungen/arbeitsbereiche/allgpsy/Download/BAWL/index.html). 
You will need to request a password from the authors. Save the file **without password** as 
`emocoder/data/datasets/Vo.csv`. We had to run an automatic file repair when opening it with Excel for the 
first time.
- de2. Get the file  `13428_2011_59_MOESM1_ESM.xls` from [Briesemeister et al. (2011)](https://doi.org/10.3758/s13428-011-0059-y) 
and save it as `emocoder/data/datasets/Briesemeister2011.xls`.
- pl1. Get the file `13428_2014_552_MOESM1_ESM.xlsx` from [Riegel et al. (2015)](https://doi.org/10.3758/s13428-014-0552-1) and 
save it as `emocoder/data/datasets/Riegel2015.xlsx`.
- pl2. Get the file `S1 Dataset` from [Wierzba et al. (2015)](https://doi.org/10.1371/journal.pone.0132305) 
and save it as `emocoder/data/datasets/Wierzba2015.xlsx`.
- tr1. Get the file `TurkishEmotionalWordNorms.csv` from [Kapucu et al. (2018)](https://doi.org/10.1177/0033294118814722) 
 which is available [here](https://osf.io/rxtdm/). Place it under `emocoder/data/datasets/Kapucu.csv`.
- tr2. This dataset is included in tr1.

### Text Datasets
- Affective Text. Get the archive `AffectiveText.Semeval.2007.tar.gz` from [Strapparava and Mihalcea (2007)](http://web.eecs.umich.edu/~mihalcea/affectivetext/)
and save it as `emocoder/data/datasets/AffectiveText.tar.gz`.
- EmoBank. This dataset will download automatically from [GitHub](https://github.com/JULIELab/EmoBank) when needed.
- CVAT. Get the archive `ChineseEmoBank.zip` from [Lung-Hao Lee](https://www.lhlee.net) and save it as 
`emocoder/data/datasets/ChineseEmoBank.zip`. We requested the dataset directly from the author via personal communication.

### Image Datasets
- Facial Emotion Recognition. Get the file `fer2013.csv"` from
    [this](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) Kaggle
    competition as described in [this](https://arxiv.org/pdf/1307.0414.pdf) paper. Also get, the file `imdb_DimEmotion.mat`
    from [this](https://github.com/aimerykong/Dimensional-Emotion-Analysis-of-Facial-Expression) repository, as 
    described in [this](https://doi.org/10.1016/j.neucom.2020.01.067) paper. Combine both files into a new file called 
    `fer2013+vad.csv` as illustrated in the notebook `emocoder/scripts/read-and-combine-FER2013-vad-data.ipynb`.
    Place this file under `emocoder/data/datasets/fer2013`.
- AffectNet. Get the AffectNet datbase from [this](https://doi.org/10.1109/TAFFC.2017.2740923) paper and place the 
    folders `Labels` and `ManuallyAnnotated`  under `emocoder/data/datasets/AffectNet`.

## Replicating the Experiments

1. From the project root, run `python emocoder/scripts/setup-target-folder-structure.py`. This will create a new `target`
folder and all necessary subfolders. If the `target` folder already exists, rename it first to keep old and new 
    results separate. 
2. `python emocoder/scripts/run_all_mapping_experiments.py` (this should only take a couple of minutes). Per default, the experiments will be run on on gpu 0. Use the `--gpu` parameter to choose a gpu.
3. Identify the path to the model checkpoint in the "no_nrc" condition. This should be something like 
    `emocoder/target/mapping/multitask/dev/no_nrc-<someTimestamp>/checkpoints/model_<epochNumber>.pt`. Insert this path 
    in `emocoder/experiments/config.json` in the "EMOTION_CODEC_PATH" field.
4. `python emocoder/scripts/run_all_baseline_experiments.py` (this may take a couple of hours)
5. `python emocoder/scripts/run_all_encoder_experiments.py` (this may take a couple of hours)
6. `python emocoder/scripts/run_all_checkpoint_test.py` (this may take a couple of hours)
7. `python emocoder/scripts/aggregate_all.py --split test`

You have now re-run all of our experiments. You can find your replicated results within the `emocoder/target/` directory.
Note that small deviations are to be expected due to inherent randomness.

The best way to inspect the results are the notebooks under `emocoder/analysis`.

To recreate our visualizations of the emotion space, `cd emocoder/analysis` and run the four `get_*.py` scripts in there:
1. `python get_prediction_head_embeddings.py`
2. `python get_word_emotion_embeddings.py`
3. `python get_text_emotion_embeddings.py`
4. `python get_image_emotion_embeddings.py`

The figures from the paper can then be accessed by running `Interlingua-Visualization.ipynb`.

Due to the randomness inherent to the training process, the plots will look slightly different than the published versions. The original plots can be recreated by
running the above steps with the original experimental results (published separately).

## Contact

Should you have any further questions, please reach out to me via sven.buechel@uni-jena.de.
