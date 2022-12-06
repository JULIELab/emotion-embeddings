"""
Loads text datasets, feeds data through respecive encoder model, stores resulting embeddings to disk.
"""

import pandas as pd
import torch

from emocoder.experiments import constants as xconstants

from emocoder.experiments.image.multitask import ImageMultitaskExperiment as experiment
from emocoder.src import data

from emocoder.src.experiments import get_best_checkpoint
from emocoder.src.utils import get_experiment_dir, get_analysis_dir
from torch.utils.data import DataLoader

import argparse

# LOADING MODELS
FER = get_best_checkpoint(get_experiment_dir(xconstants.IMAGE_MULTITASK_BASEPATH / "dev", "affectnet_multitask"))
AFFECTNET = get_best_checkpoint(get_experiment_dir(xconstants.IMAGE_MULTITASK_BASEPATH / "dev", "fer_multitask"))

checkpoint_fer = torch.load(FER)
checkpoint_affnet = torch.load(AFFECTNET)

model_fer, __ = experiment.get_model()
model_affnet, __ = experiment.get_model()

model_fer.load_state_dict(checkpoint_fer, strict=False)
model_affnet.load_state_dict(checkpoint_affnet, strict=False)

transform = data.images.get_ResNet_Preprocessor(data_augmentation=False)


# LOADING DATA, ...
ds_fer = data.images.FER2013("dev", transform=transform)
ds_affnet = data.images.AffectNet2019_BE("dev", transform=transform) # full set is too large I guess


dl_fer = DataLoader(dataset=ds_fer,
                        batch_size=32,
                        shuffle=False,
                        num_workers=0,)
dl_affnet = DataLoader(dataset=ds_affnet,
                        batch_size=32,
                        shuffle=False,
                        num_workers=0,)




def get_emotion_embeddings(model, dataloader, gpu):
    device = torch.device(f"cuda:{gpu}")
    emb_list = []
    id_list = []
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            if i_batch % 100 == 0:
                print(f"Currently processsing batch {i_batch} of {len(dataloader)}.")
            indx = batch["id"].tolist()
            features = batch["features"].to(device)
            embs = model.enc(x=features, encoder="images")
            id_list += indx
            emb_list.append(embs)
    embs = torch.cat(emb_list).cpu().numpy()
    return id_list, embs




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes and stores emotion embedding for image dataesets FER "
                                                 "and AFFECTNET")
    parser.add_argument("--gpu", help="Which gpu to ues", default=0, type=int)

    args = parser.parse_args()


    for name, model, dl in (
            ("fer", model_fer, dl_fer),
            ("affectnet", model_affnet, dl_affnet)):
        print(name)
        ids, embs = get_emotion_embeddings(model, dl, args.gpu)
        df = pd.DataFrame(index=ids, data=embs)
        df.to_csv(get_analysis_dir() / f"image_emotion_embeddings_{name}.csv")