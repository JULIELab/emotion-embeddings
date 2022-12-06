"""
Loads text datasets, feeds data through respecive encoder model, stores resulting embeddings to disk.
"""

import pandas as pd
import torch

from emocoder.experiments import constants as xconstants

from emocoder.experiments.text.multitask import Exp as experiment
from emocoder.src import data

from emocoder.src.experiments import get_best_checkpoint
from emocoder.src.utils import get_experiment_dir, get_analysis_dir
from torch.utils.data import DataLoader

# LOADING MODELS
EN = get_best_checkpoint(get_experiment_dir(xconstants.TEXT_MULTITASK_BASEPATH / "dev", "emobank-afftext"))
ZH = get_best_checkpoint(get_experiment_dir(xconstants.TEXT_AUGMENTED_BASEPATH / "dev", "cvat"))

checkpoint_en = torch.load(EN)
checkpoint_zh = torch.load(ZH)

model_en, tokenizer_en, __ = experiment.get_model("bert-base-uncased")
model_zh, tokenizer_zh, __ = experiment.get_model("bert-base-chinese")

transform_en = data.utils.Tokenize_Transformer(tokenizer=tokenizer_en)
transform_zh = data.utils.Tokenize_Transformer(tokenizer=tokenizer_zh)

model_en.load_state_dict(checkpoint_en, strict=False)
model_zh.load_state_dict(checkpoint_zh, strict=False)


# LOADING DATA, ...
ds_emobank = data.text.EmoBank("full", transform=transform_en)
ds_afft = data.text.AffTextBE5("full", transform=transform_en)
ds_sst = data.text.SST_2_Class("full", transform=transform_en)
ds_cvat = data.text.CVAT("full", transform=transform_zh)

collater_emobank = data.utils.Collater(padding_symbol=tokenizer_en.pad_token_id,
                                       num_labels=len(ds_emobank.variables),
                                       label_dtype=torch.float32)
collater_afft = data.utils.Collater(padding_symbol=tokenizer_en.pad_token_id,
                                    num_labels=len(ds_afft.variables),
                                    label_dtype=torch.float32)
collater_sst = data.utils.Collater(padding_symbol=tokenizer_en.pad_token_id,
                                   num_labels=len(ds_sst.variables),
                                   label_dtype=torch.float32)
collater_cvat = data.utils.Collater(padding_symbol=tokenizer_zh.pad_token_id,
                                    num_labels=len(ds_cvat.variables),
                                    label_dtype=torch.float32)

dl_emobank = DataLoader(dataset=ds_emobank,
                        batch_size=32,
                        shuffle=False,
                        num_workers=16,
                        collate_fn=collater_emobank)
dl_afft = DataLoader(dataset=ds_afft,
                     batch_size=32,
                     shuffle=False,
                     num_workers=16,
                     collate_fn=collater_afft)
dl_sst = DataLoader(dataset=ds_sst,
                    batch_size=32,
                    shuffle=False,
                    num_workers=16,
                    collate_fn=collater_sst)
dl_cvat = DataLoader(dataset=ds_cvat,
                     batch_size=32,
                     shuffle=False,
                     num_workers=16,
                     collate_fn=collater_cvat)



def get_emotion_embeddings(model, dataloader, device=torch.device("cuda:0")):
    emb_list = []
    text_list = []
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            text = batch["raw"]
            features = batch["features"].to(device)
            embs = model.enc(x=features, encoder="text")
            text_list += text
            emb_list.append(embs)
    embs = torch.cat(emb_list).cpu().numpy()
    return text_list, embs

for name, model, dl in (
        ("emobank", model_en, dl_emobank),
        ("afftext", model_en, dl_afft),
        ("sst", model_en, dl_sst),
        ("cvat", model_zh, dl_cvat)
        ):
    print(name)
    texts, embs = get_emotion_embeddings(model, dl)
    df = pd.DataFrame(index=texts, data=embs)
    df.to_csv(get_analysis_dir() / f"text_emotion_embeddings_{name}.csv")