from emocoder.src.models import NumericToDiscreteDecoder
from emocoder.src.data import words
from emocoder.src.data.utils import MinMaxScaler
from emocoder.src import models
from emocoder.src import utils
import torch
import pandas as pd
from emocoder.experiments.mapping import multitask as multitask_mapping_experiment
from emocoder.experiments import constants as xconstants
import logging
from emocoder.src.utils import get_experiment_dir
from emocoder.src.experiments import get_best_checkpoint
import json


class VADBE5_to_Class_Decoder(NumericToDiscreteDecoder):

    def __init__(self,
                 primary_decoders,
                 classes): #list of strings to look up in Warriner + Warriner_BE

        # load warriner + warriner be
        vad_scaler = MinMaxScaler(1, 9, -1, 1)
        be5_scaler = MinMaxScaler(1,5, 0, 1)

        df_vad = words.XANEW.get_df()
        df_vad = df_vad.applymap(vad_scaler)

        df_be5 = words.XANEW_BE.get_df()
        df_be5 = df_be5.applymap(be5_scaler)

        df = pd.concat([df_vad, df_be5], axis=1)
        matrix = torch.tensor(df.loc[classes].values.T)

        super().__init__(primary_decoders=primary_decoders, mapping_matrix=matrix)


class VAD_to_Class_Decoder(NumericToDiscreteDecoder):

    def __init__(self,
                 primary_decoders,
                 classes):  # list of strings to look up in Warriner + Warriner_BE

        # load warriner + warriner be
        scaler = MinMaxScaler(1, 9, -1, 1)

        df = words.XANEW.get_df()
        df = df.applymap(scaler)

        matrix = torch.tensor(df.loc[classes].values.T)

        super().__init__(primary_decoders=primary_decoders, mapping_matrix=matrix)


def get_pretrained_emotion_codec():

    # load config file
    experiments_config = get_experiments_config()
    emotion_codec_path = experiments_config["EMOTION_CODEC_PATH"]
    emotion_codec_architecture = experiments_config["EMOTION_CODEC_ARCHITECTURE"]

    # load model
    model = multitask_mapping_experiment.EXPERIMENTS[emotion_codec_architecture].get_model()
    CHECKPOINT = utils.get_project_root() / emotion_codec_path
    logging.info(f"Loading checkpoint from {CHECKPOINT}.")
    state_dict = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)

    #adding new secondary decoder for SST
    model.dec["pol1"] = models.SubsetDecoder(primary_decoder=model.dec["vad"], subset=[0])


    return model, state_dict

def get_experiments_config():
    with open(utils.get_project_root() / "emocoder" / "experiments" / "config.json") as f:
        config = json.load(f)
    return config


