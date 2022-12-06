import numpy as np
import pandas as pd
from zipfile import ZipFile
from pathlib import Path
from emocoder.src import data, utils
from sklearn.model_selection import train_test_split
import json
from urllib.request import urlretrieve



split_dir = utils.get_data_dir() / "splits"


def split_ssec():
    # splitting SSEC test into test and dev (the authors provide train and test split but no dev set. Splitting the
    # test set 50-50 gives roughly a 3-1-1 train-dev-test split, whereas splitting the train would make for a much to small
    # train set in comparison to the dev set. The authors did only use non-contextualized models, so exact comparability is
    # not super important here.
    path = utils.get_dataset_dir() / "SSEC.zip"
    with ZipFile(path) as zip:
        with zip.open("ssec-aggregated/test-combined-0.0.csv") as f:
            df = pd.read_csv(f, sep="\t",names= ["Joy", "Anger", "Sadness", "Fear", "Disgust", "Surprise", "Anticipation", "Trust"] + ["Text"])
    # print(df.head())

    dev, test = train_test_split(df.index, random_state=42, test_size=.5)
    splits = {'dev': list(dev), "test":list(test)}
    with open(split_dir / "SSEC_splits.json", "w") as f:
        json.dump(splits, f)



def split_ISEAR():
    # Splitting ISEAR
    url = "https://github.com/JULIELab/ISEAR/archive/master.zip"
    path = Path.home()/"data/isear.zip"
    if not path.is_file():
        print('Downloading data...')
        urlretrieve(url, path)
    with ZipFile(path) as zip:
        with zip.open("ISEAR-master/isear.csv") as f:
            df = pd.read_csv(f, index_col="MYKEY")[["Field1", "SIT"]]

    df.columns = ["label", "text"]

    def keep(x):
        x = x.strip().lower()
        exclude = ["forced to fill in a questionnaire.",
                   "none.",
                   "nothing.",
                   "no response.",
                   "not applicable to myself.",
                   "doesn't apply.",
                   "can't remember having had this feeling",
                   "can't remember that feeling",
                   "does not apply",
                   'the same as in "shame".',
                   "not included on questionnaire",
                   "Not applicable."]
        exclude = [s.strip().lower() for s in exclude]
        if x.startswith("[") and x.endswith("]") or x in exclude:
            return False
        else:
            return True

    df =df[df.text.apply(keep)]

    train, tmp = train_test_split(df.index, random_state=42, train_size=.8)
    dev, test = train_test_split(tmp, random_state=42, train_size=.5)

    # make sure that there is no overlap between the sets
    for set1, set2 in [(train, dev), (train, test), (dev, test)]:
        assert len(set(set1).intersection((set(set2)))) == 0

    splits = {"train": list(train), "dev": list(dev), "test": list(test)}
    with open(split_dir / "ISEAR_splits.json", "w") as f:
        json.dump(splits, f)


def split_NRC():

    df = data.words.NRC.get_df()

    train, tmp = train_test_split(df.index, random_state=1, train_size=.8)
    dev, test = train_test_split(tmp, random_state=1, train_size=.5)

    # make sure that there is no overlap between the sets
    for set1, set2 in [(train, dev), (train, test), (dev, test)]:
        assert len(set(set1).intersection((set(set2)))) == 0

    splits = {"train": list(train), "dev": list(dev), "test": list(test)}

    with open(split_dir / "NRC_splits.json", "w") as f:
        json.dump(splits, f)


def split_Moors2013():

   df = data.words.Moors2013.get_df()

   train, tmp = train_test_split(df.index, random_state=1, train_size=.8)
   dev, test = train_test_split(tmp, random_state=1, train_size=.5)

   # make sure that there is no overlap between the sets
   for set1, set2 in [(train, dev), (train, test), (dev, test)]:
       assert len(set(set1).intersection((set(set2)))) == 0

   splits = {"train": list(train), "dev": list(dev), "test": list(test)}

   with open(split_dir / "Moors2013_splits.json", "w") as f:
       json.dump(splits, f)



def split_IAPS():
    dataset = data.images.IAPS2008(split="full")._df

    train, tmp = train_test_split(dataset.index, random_state=1, train_size=.8)
    dev, test = train_test_split(tmp, random_state=1, train_size=.5)

    # make sure that there is no overlap between the sets
    for set1, set2 in [(train, dev), (train, test), (dev, test)]:
        assert len(set(set1).intersection((set(set2)))) == 0

    splits = {"train": list(train), "dev": list(dev), "test": list(test)}

    with open(split_dir / "IAPS2008_splits.json", "w") as f:
        json.dump(splits, f)

def split_YOU2017():
    dataset = data.images.YOU2017(split="full")

    train, tmp = train_test_split(list(range(0, len(dataset))), random_state=1, train_size=.8)
    dev, test = train_test_split(tmp, random_state=1, train_size=.5)

    # make sure that there is no overlap between the sets
    for set1, set2 in [(train, dev), (train, test), (dev, test)]:
        assert len(set(set1).intersection((set(set2)))) == 0

    splits = {"train": list(train), "dev": list(dev), "test": list(test)}

    with open(split_dir / "YOU2017_splits.json", "w") as f:
        json.dump(splits, f)


def split_FER2013():
    dataset = data.images.FER2013(split="full")

    train, tmp = train_test_split(list(range(0, len(dataset))), random_state=1, train_size=.8)
    dev, test = train_test_split(tmp, random_state=1, train_size=.5)

    # make sure that there is no overlap between the sets
    for set1, set2 in [(train, dev), (train, test), (dev, test)]:
        assert len(set(set1).intersection((set(set2)))) == 0

    splits = {"train": list(train), "dev": list(dev), "test": list(test)}

    with open(split_dir / "FER2013_splits.json", "w") as f:
        json.dump(splits, f)


def split_df(df, key, seed, ratio = (8,1,1)):
    sum = np.sum(ratio)
    train_ratio, dev_ratio, test_ratio = ratio[0]/sum, ratio[1]/sum, ratio[2]/sum
    train, tmp = train_test_split(df.index, random_state=seed, train_size=train_ratio)
    dev, test = train_test_split(tmp, random_state=seed+1, train_size=.5)

    # make sure that there is no overlap between the sets
    for set1, set2 in [(train, dev), (train, test), (dev, test)]:
        assert len(set(set1).intersection((set(set2)))) == 0

    splits = {"train": list(train), "dev": list(dev), "test": list(test)}

    with open(split_dir / f"{key}_splits.json", "w") as f:
        json.dump(splits, f)

def split_Anew_Stevenson():
    anew = data.words.ANEW1999.get_df()
    steve = data.words.Stevenson2007.get_df()
    common = sorted(list(set(anew.index).intersection(set(steve.index))))

    ratio = (3,1,1)
    sum = np.sum(ratio)
    train_ratio, dev_ratio, test_ratio = ratio[0] / sum, ratio[1] / sum, ratio[2] / sum
    train, tmp = train_test_split(common, random_state=1, train_size=train_ratio)
    dev, test = train_test_split(tmp, random_state=2 , train_size=.5)

    # make sure that there is no overlap between the sets
    for set1, set2 in [(train, dev), (train, test), (dev, test)]:
        assert len(set(set1).intersection((set(set2)))) == 0

    splits = {"train": list(train), "dev": list(dev), "test": list(test)}
    with open(split_dir / "ANEW-Stevenson_splits.json", "w") as f:
        json.dump(splits, f)

def split_XANEW_NRC():

    xanew = data.words.XANEW.get_df()
    nrc = data.words.NRC.get_df()
    common = sorted(list(set(xanew.index).intersection(set(nrc.index))))

    ratio = (3,1,1)
    sum = np.sum(ratio)
    train_ratio, dev_ratio, test_ratio = ratio[0] / sum, ratio[1] / sum, ratio[2] / sum
    train, tmp = train_test_split(common, random_state=11, train_size=train_ratio)
    dev, test = train_test_split(tmp, random_state=12 , train_size=dev_ratio / (dev_ratio + test_ratio))

    # make sure that there is no overlap between the sets
    for set1, set2 in [(train, dev), (train, test), (dev, test)]:
        assert len(set(set1).intersection((set(set2)))) == 0

    # make sure all sets combined equal length common index
    assert common == sorted(list(set(train).union(set(dev)).union(set(test))))

    splits = {"train": list(train), "dev": list(dev), "test": list(test)}
    with open(split_dir / "XANEW-NRC_splits.json", "w") as f:
        json.dump(splits, f)

def split_AffText():
    """
    New datasplit for transfer learning experiments from EmoBank. EmoBank in part contains sentences from AffText,
    annotated not with BE6 but with VAD. The new datasplit makes sure that information cannot leak from, e.g.,
    EmoBank train splits to AffText test splits by using the same train-dev-test assingmet for AffText as for EmoBank.
    Some of the annotated sentences (58) are lost in the process, though. They were removed from EmoBank as part of the
    data cleansing procedure (Buechel & Hahn, EACL 2017).
    """

    # Identify AffText part in EmoBank
    eb = data.text.EmoBank.get_df()
    splitting_index = [id.split("_") for id in eb.index]
    query = [True if parts[0]=="SemEval" else False for parts in splitting_index ]
    eb_AffText = eb[query]

    # Add instances to train-dev-test splits according to EmoBank assignment
    splits = {"train": [], "dev": [], "test": []}
    for index, row in eb_AffText.iterrows():
        __, i = index.split("_")
        eb_split = row.loc["split"]
        splits[eb_split].append(int(i))

    for key, ids in splits.items():
        splits[key] = sorted(ids)

    # print(*[len(splits[part]) for part in ["train", "dev", "test"]])
    with open(split_dir / "AffTextEB_splits.json", "w") as f:
        json.dump(splits, f)



if __name__=="__main__":
    print(f"Writing datasets splits in directory {split_dir}...")

    split_ISEAR()
    split_ssec()
    split_NRC()
    split_IAPS()
    split_YOU2017()
    split_FER2013()
    split_Moors2013()
    split_df(data.words.Moors2013.get_df(), "Moors2013", 1, ratio = (3, 1, 1))
    split_df(data.words.Stadthagen_VA.get_df(), "StadthagenVA", 375243454, ratio = (8, 1, 1))
    split_df(data.words.Stadthagen_BE.get_df(), "StadthagenBE", 999, ratio = (8, 1, 1))
    split_df(data.words.Imbir.get_df(), "Imbir", 1, ratio=(3, 1, 1))
    split_df(data.words.Vo.get_df(), "Vo", 11, ratio=(3, 1, 1))
    split_df(data.words.Briesemeister.get_df(), "Briesemeister", 42, ratio=(3, 1, 1))
    split_df(data.words.Riegel.get_df(), "Riegel", 3452562, ratio=(3, 1, 1))
    split_df(data.words.Wierzba.get_df(), data.words.Wierzba.split_key, 246243, (3, 1, 1))
    split_df(data.words.Kapucu.get_df(), "Kapucu", 111, ratio=(3,1,1))
    split_Anew_Stevenson()
    split_XANEW_NRC()
    split_AffText()

    print("Dataset splits written!")


