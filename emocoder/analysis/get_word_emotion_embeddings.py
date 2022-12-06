"""
Loads one word dataset per language, feeds each of the datasets through the respective encoder model (needs to
load the respective word embedding model for doing so), and stores the resulting emotion embeddings in csv files.
"""

import pandas as pd
import seaborn as sns
sns.set(style="white")
import torch

from emocoder.experiments import utils as xutils, constants as xconstants


from emocoder.experiments.word.multitask import Fit_Word_Encoders_on_Multiple_Datasets as experiment
from emocoder.src.data.words import ANEW1999,Stadthagen_VA, Vo, Riegel, Kapucu_VA
from emocoder.src import data

from emocoder.src.experiments import get_best_checkpoint
from emocoder.src.utils import get_experiment_dir, get_analysis_dir

model, state_dict = xutils.get_pretrained_emotion_codec()


EN = get_best_checkpoint(get_experiment_dir(xconstants.WORD_MULTITASK_BASEPATH/ "dev", "Fit-Word-Encoder-on-English"))
ES = get_best_checkpoint(get_experiment_dir(xconstants.WORD_MULTITASK_BASEPATH / "dev", "Fit-Word-Encoder-on-Spanish"))
DE = get_best_checkpoint(get_experiment_dir(xconstants.WORD_MULTITASK_BASEPATH / "dev", "Fit-Word-Encoder-on-German"))
PL = get_best_checkpoint(get_experiment_dir(xconstants.WORD_MULTITASK_BASEPATH / "dev", "Fit-Word-Encoder-on-Polish"))
TR = get_best_checkpoint(get_experiment_dir(xconstants.WORD_MULTITASK_BASEPATH / "dev", "Fit-Word-Encoder-on-Turkish"))




checkpoint_en = torch.load(EN)
checkpoint_es = torch.load(ES)
checkpoint_de = torch.load(DE)
checkpoint_pl = torch.load(PL)
checkpoint_tr = torch.load(TR)

model_en, __ = experiment.get_model()
model_es, __ = experiment.get_model()
model_de, __ = experiment.get_model()
model_pl, __ = experiment.get_model()
model_tr, __ = experiment.get_model()

model_en.load_state_dict(checkpoint_en, strict=False)
model_es.load_state_dict(checkpoint_es, strict=False)
model_de.load_state_dict(checkpoint_de, strict=False)
model_pl.load_state_dict(checkpoint_pl, strict=False)
model_tr.load_state_dict(checkpoint_tr, strict=False)

transform_en = data.utils.Embedding_Lookup_Transform(embeddings="FB_CC_EN", limit=None)
transform_es = data.utils.Embedding_Lookup_Transform(embeddings="FB_CC_ES", limit=None)
transform_de = data.utils.Embedding_Lookup_Transform(embeddings="FB_CC_DE", limit=None)
transform_pl = data.utils.Embedding_Lookup_Transform(embeddings="FB_CC_PL", limit=None)
transform_tr = data.utils.Embedding_Lookup_Transform(embeddings="FB_CC_TR", limit=None)

ds_en = ANEW1999(split="full", transform=transform_en)
ds_es = Stadthagen_VA(split="full", transform=transform_es)
ds_de = Vo(split="full", transform=transform_de)
ds_pl = Riegel(split="full", transform=transform_pl)
ds_tr = Kapucu_VA(split="full", transform=transform_tr)

# hacky repair of turkish dataset
ds_tr.df.rename(inplace=True, index={"teröri̇st":"terörist", "i̇yi̇leşmek": "iyileşmek"})

def get_emotion_embeddings(model, dataset, device=torch.device("cuda:0")):
    emb_list = []
    word_list = []
    dl = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=128)
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        for batch in dl:
            words = batch["raw"]
            features = batch["features"].to(device)
            embs = model.enc(x=features, encoder="words")
            word_list += words
            emb_list.append(embs)
    embs = torch.cat(emb_list).cpu().numpy()
    return word_list, embs

words_en, embs_en = get_emotion_embeddings(model_en, ds_en)
words_es, embs_es = get_emotion_embeddings(model_es, ds_es)
words_de, embs_de = get_emotion_embeddings(model_de, ds_de)
words_pl, embs_pl = get_emotion_embeddings(model_pl, ds_pl)
words_tr, embs_tr = get_emotion_embeddings(model_tr, ds_tr)


df_en = pd.DataFrame(index=words_en, data=embs_en)
df_es = pd.DataFrame(index=words_es, data=embs_es)
df_de = pd.DataFrame(index=words_de, data=embs_de)
df_pl = pd.DataFrame(index=words_pl, data=embs_pl)
df_tr = pd.DataFrame(index=words_tr, data=embs_tr)

for lang, df in (("en", df_en), ("es", df_es), ("de", df_de),  ("pl", df_pl), ("tr", df_tr)):
    df.to_csv(get_analysis_dir() / f"word_emotion_embeddings_{lang}.csv")