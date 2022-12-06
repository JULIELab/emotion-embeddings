from emocoder.src import data
import torch
import numpy as np
from torch.utils.data import DataLoader
from random import randint
from emocoder.src.utils import  get_split


class Dummy_Tokenizer():

    def __init__(self, lang="en"):
        self.lang = lang

    def encode(self, s):
        if self.lang == "en":
            tokens = s.split()
        elif self.lang=="zh":
            tokens = list(s)
        indices = [randint(0, 1000) for t in tokens]
        return indices




# ======= TEXT ======== #
def test_EmoBank():
    transform = data.utils.Tokenize_Transformer(tokenizer=Dummy_Tokenizer(), label_to_numpy=False)
    for split in "train", "dev", "test":
        ds = data.text.EmoBank(split, transform=transform)
        dl = DataLoader(dataset=ds, batch_size=16, shuffle=True, collate_fn=data.utils.Collater(0, 1, torch.int64))
        for batch in dl:
            pass


def test_SST_5_Class():
    train, dev, test = [data.text.SST_5_Class(split) for split in ("train", "dev", "test")]
    assert len(data.text.SST_5_Class(split="train")) == 8544
    assert len(data.text.SST_5_Class(split="dev")) == 1101
    assert len(data.text.SST_5_Class(split="test")) == 2210

    sample = train[0]

    assert isinstance(sample["raw"], str)
    assert isinstance(sample["label"], int) and sample["label"] >= 0 and sample["label"] <= 4

    train = data.text.SST_5_Class(split="train", transform=data.utils.Tokenize_Transformer(tokenizer=Dummy_Tokenizer(),
                                                                                label_to_numpy=False))
    x = train[0]

    loader = DataLoader(dataset=train, batch_size=12, shuffle=True, collate_fn=data.utils.Collater(padding_symbol=0,
                                                                                             num_labels=1,
                                                                                             label_dtype=torch.int64))
    for batch in loader:
        features, labels = batch["features"], batch["labels"]
        assert features.dtype == torch.int64
        assert features.ndim == 2
        assert labels.dtype == torch.int64
        assert labels.ndim == 1
        break


def test_SST_2_Class():
    Dsc = data.text.SST_2_Class

    split_sizes = {
        "train": 6920,
        "dev": 871,
        "test":1821,
    }

    transform = data.utils.Tokenize_Transformer(tokenizer=Dummy_Tokenizer(), label_to_numpy=False)
    for split in ["train", "dev", "test"]:
        ds =Dsc(split=split, transform=transform)
        assert len(ds) == split_sizes.get(split)
        dl = DataLoader(dataset=ds,
                        batch_size=100,
                        collate_fn=data.utils.Collater(0, 1, torch.int64),
                        shuffle=True)
        for batch in dl:
            pass



def test_ISEAR():
    transform = data.utils.Tokenize_Transformer(tokenizer=Dummy_Tokenizer(), label_to_numpy=False)
    for split in ["train", "dev", "test"]:
        ds =  data.text.ISEAR(split=split, transform=transform)
        dl = DataLoader(dataset=ds,
                        batch_size=100,
                        collate_fn=data.utils.Collater(0, 1, torch.int64),
                        shuffle=True)
        for batch in dl:
            pass


def test_SSEC():

    tokenizer = Dummy_Tokenizer(lang="en")
    preprocessor = data.utils.Tokenize_Transformer(tokenizer=tokenizer, label_to_numpy=True)
    collater = data.utils.Collater(padding_symbol=0, num_labels=8, label_dtype=torch.int32)

    for split in "train", "dev", "test":
        ds = data.text.SSEC(split, transform=preprocessor)
        dl = DataLoader(dataset=ds, batch_size=12, shuffle=True, collate_fn=collater)
        for batch in dl:
            pass


def test_CVAT():

    tokenizer = Dummy_Tokenizer(lang="zh")
    preprocessor = data.utils.Tokenize_Transformer(tokenizer=tokenizer, label_to_numpy=True)
    collater = data.utils.Collater(padding_symbol=0, num_labels=2, label_dtype=torch.float32)
    for split in "train", "dev", "test":
        ds = data.text.CVAT(split, transform=preprocessor)
        dl = DataLoader(dataset=ds, batch_size=12, shuffle=True, collate_fn=collater)
        for batch in dl:
            pass


def test_AffText():
    tokenizer = Dummy_Tokenizer(lang="en")
    preprocessor = data.utils.Tokenize_Transformer(tokenizer=tokenizer, label_to_numpy=True)
    collater = data.utils.Collater(padding_symbol=0, num_labels=6, label_dtype=torch.float32)
    for split in "train", "dev", "test":
        ds = data.text.AffText(split, transform=preprocessor)
        dl = DataLoader(dataset=ds, batch_size=12, shuffle=True, collate_fn=collater)
        for batch in dl:
            pass

# ======= Words ======== #

# def test_XANEW():
#     emb_transform = data.utils.Embedding_Lookup_Transform(embeddings="FB_CC_EN", limit=10 * 1000)
#     for split in ["train", "dev", "test"]:
#         ds = data.words.XANEW(split, emb_transform)
#         dl = DataLoader(ds, batch_size=128, shuffle=True)
#         for batch in dl:
#             pass

# def test_XANEW_BE():
#     emb_transform = data.utils.Embedding_Lookup_Transform(embeddings="FB_CC_EN", limit=10 * 1000)
#     for split in ["train", "dev", "test"]:
#         ds = data.words.XANEW_BE(split, emb_transform)
#         dl = DataLoader(ds, batch_size=128, shuffle=True)
#         for batch in dl:
#             pass



def test_ANEW1999():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings="FB_CC_EN", limit=10*1000)
    for split in ["train", "dev", "test"]:
        dataset = data.words.ANEW1999(split=split, transform=emb_transform)
        dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
        for batch in dataloader:
            assert np.sum(np.abs(batch['features'].numpy())) > 0

def test_Stevenson2007():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_EN', limit=10*1000)
    for split in ["train", "dev", "test"]:
        ds = data.words.Stevenson2007(split=split, transform=emb_transform)
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            assert np.sum(np.abs(batch['features'].numpy())) > 0

def test_NRC():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_EN', limit=1000)
    for split in ["train", "dev", "test"]:
        ds = data.words.NRC(split, emb_transform)
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass

def test_Stadthagen_VA():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_ES', limit=1000)
    splits = get_split("StadthagenVA")
    for split in ["train", "dev", "test"]:
        ds = data.words.Stadthagen_VA(split=split, transform=emb_transform)
        assert len(ds) == len(splits[split])
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass


def test_Stadthagen_BE():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_ES', limit=1000)
    splits = get_split("StadthagenBE")
    for split in ["train", "dev", "test"]:
        ds = data.words.Stadthagen_BE(split=split, transform=emb_transform)
        assert len(ds) == len(splits[split])
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass


def test_Imbir():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_PL', limit=1000)
    splits = get_split("Imbir")
    for split in ["train", "dev", "test"]:
        ds = data.words.Imbir(split=split, transform=emb_transform)
        assert len(ds) == len(splits[split])
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass


def test_Vo():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_DE', limit=10*1000)
    splits = get_split("Vo")
    for split in ["train", "dev", "test"]:
        ds = data.words.Vo(split=split, transform=emb_transform)
        assert len(ds) == len(splits[split])
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass

def test_Biesemeister():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_DE', limit=10*1000)
    splits = get_split("Briesemeister")
    for split in ["train", "dev", "test"]:
        ds = data.words.Briesemeister(split=split, transform=emb_transform)
        assert len(ds) == len(splits[split])
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass


def test_Riegel():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_PL', limit=10*1000)
    splits = get_split("Riegel")
    for split in ["train", "dev", "test"]:
        ds = data.words.Riegel(split=split, transform=emb_transform)
        assert len(ds) == len(splits[split])
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass


def test_Wierzba():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_PL', limit=10*1000)
    splits = get_split("Wierzba")
    for split in ["train", "dev", "test"]:
        ds = data.words.Wierzba(split=split, transform=emb_transform)
        assert len(ds) == len(splits[split])
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass

def test_Kapucu_base():
    df = data.words.Kapucu.get_df()
    pass

def test_Kapucu_VA():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_TR', limit=10*1000)
    splits = get_split("Kapucu")

    # testing splits
    for split in ["train", "dev", "test"]:
        ds = data.words.Kapucu_VA(split=split, transform=emb_transform)
        assert len(ds) == len(splits[split])
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass

def test_Kapucu_BE():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_TR', limit=10*1000)
    splits = get_split("Kapucu")
    for split in ["train", "dev", "test"]:
        ds = data.words.Kapucu_BE(split=split, transform=emb_transform)
        assert len(ds) == len(splits[split])
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass

def test_Moors2013():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings='FB_CC_NL', limit=1000)
    for split in ["train", "dev", "test"]:
        ds = data.words.Moors2013(split, emb_transform)
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass


# ==========  Mapping ================ #
def test_FER_BE_VAD():
    ds = data.mapping.FER_BE_VAD("full")
    dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
    for batch in dl:
        pass


def test_ANEW_Stevenson():
    for split in ["train", "dev", "test"]:
        ds = data.mapping.ANEW_Stevenson(split)
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass

def test_XANEW_NRC():
    for split in ["train", "dev", "test"]:
        ds = data.mapping.XANEW_NRC(split)
        dl = DataLoader(dataset=ds, batch_size=128, shuffle=True)
        for batch in dl:
            pass

def test_AffectNet_Mapping():
    for spl in ["train", "dev", "test"]:
        features, labels = data.mapping.AffectNet_Mapping.load_data(split=spl)
        assert len(features) == len(labels)




# ======= Images ======== #

def test_YOU2017():
    assert len(data.images.YOU2017("full")) == 23185
    assert len(data.images.YOU2017('train')) == 18548
    assert len(data.images.YOU2017('dev')) == 2318
    assert len(data.images.YOU2017('test')) == 2319

    ds = data.images.YOU2017(split="train", transform=data.images.get_ResNet_Preprocessor(data_augmentation=True))
    dl = DataLoader(dataset=ds, batch_size=32, shuffle=True)
    for batch in dl:
        break


def test_FER2013():
    dataset = data.images.FER2013("full")
    # assert len(dataset) == 35887
    # assert len(data.FER2013('train')) == 28709
    # assert len(data.FER2013('dev')) == 3589
    # assert len(data.FER2013('test')) == 3589

    ### I am slightly subsampling the dataset to match the VAD version of if


    ds = data.images.FER2013("dev", transform=data.images.get_ResNet_Preprocessor(data_augmentation=True))
    dl = DataLoader(dataset=ds, batch_size=32, shuffle=True)
    for batch in dl:
        break

def test_IAPS2008():
    dataset = data.images.IAPS2008("full")
    assert len(dataset) == 1194
    assert len(data.images.IAPS2008('train')) == 955
    assert len(data.images.IAPS2008('dev')) == 119
    assert len(data.images.IAPS2008('test')) == 120

    ds = data.images.IAPS2008("train", transform=data.images.get_ResNet_Preprocessor(data_augmentation=True))
    dl = DataLoader(dataset=ds, batch_size=32, shuffle=True)
    for batch in dl:
        break

def test_AffectNet2019_BE():
    ds = data.images.AffectNet2019_BE("test", transform=data.images.get_ResNet_Preprocessor(data_augmentation=True))
    # running the full dataset takes waaay too long
    dl = DataLoader(dataset=ds, batch_size=32, shuffle=True, num_workers=16)
    for batch in dl:
        pass

def test_AffectNet2019_VA():
    ds = data.images.AffectNet2019_VA("test", transform=data.images.get_ResNet_Preprocessor(data_augmentation=True))
    # running the full dataset takes waaay too long
    dl = DataLoader(dataset=ds, batch_size=32, shuffle=True, num_workers=16)
    for batch in dl:
        pass

# ======= Audio? ======== #