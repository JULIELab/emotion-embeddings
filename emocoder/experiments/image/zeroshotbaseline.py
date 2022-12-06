import logging
from typing import Type

import torch
from torch import nn
from torch.utils.data import DataLoader
from emocoder.src import utils, data, models, metrics, constants
from emocoder.src.experiments import Experiment
from emocoder.experiments.image.baseline import ImageBaseline as ImageBaselineExperiment
from emocoder.experiments.mapping.baseline import MappingBaseline as MappingBaselineExperiment
import emocoder.experiments.constants as xconstants
from emocoder.experiments.experiment_classes import Checkpoint_Test_Image
from emocoder.src.experiments import get_best_checkpoint
from emocoder.src.utils import get_experiment_dir
from pathlib import Path

import argparse


class Exp(Experiment):

    def __init__(self,
                name,
                parent_dir,
                dataset_class: data.images.ImageDataset.__class__,
                image_checkpoint: Path,
                intermediate_format,
                mapping_checkpoint: Path,
                batch_size,
                split):

        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)

        self.dataset_class = dataset_class
        self.image_checkpoint = image_checkpoint
        self.mapping_checkpoint = mapping_checkpoint
        self.intermediate_format = intermediate_format
        self.batch_size = batch_size
        self.split = split


    @staticmethod
    def get_model(num_intermediate,
                  num_outputs,
                  output_scaling):
        image_model = ImageBaselineExperiment.get_model(n_outputs=num_intermediate)
        mapping_model = MappingBaselineExperiment.get_model(num_inputs=num_intermediate,
                                                            num_outputs=num_outputs,
                                                            scaling=output_scaling)
        composite_model = nn.Sequential(image_model,mapping_model)
        return composite_model


    def run(self, gpu=0):
        self.setup()
        gpu = torch.device(f"cuda:{gpu}")

        logging.info("Building model...")
        model = self.get_model(num_intermediate=len(constants.VARIABLES[self.intermediate_format]),
                               num_outputs=len(self.dataset_class.variables),
                               output_scaling="logits") #TODO Why "logits"? It is like that in the zeroshotbaseline experiments for the other experiments, too. But shouldnt this be dependend on the label format??
        model.to(device=gpu)
        logging.info("Loading model checkpoints...")
        model[0].load_state_dict(torch.load(self.image_checkpoint))
        model[1].load_state_dict(torch.load(self.mapping_checkpoint))

        test_transform = data.images.get_ResNet_Preprocessor(data_augmentation=False)


        logging.info("Start testing (there is no training phase in this experiment)")
        metric, test_loss = self.dataset_class.score(model,
                                                     device=gpu,
                                                     split=self.split,
                                                     transform=test_transform,
                                                     batch_size=self.batch_size)
        result = metric.result()

        logging.info(f"Result obtained in zero-shot scenario: {result}")
        self.save_results("zeroshot_perf", result)
        self.end_experiment()

PARENTDIR = xconstants.IMAGE_ZEROSHOTBASELINE_BASEPATH / "dev"
TESTDIR  = xconstants.IMAGE_ZEROSHOTBASELINE_BASEPATH/ "test"
IMAGE_PRETRAINED_DIR = xconstants.IMAGE_BASELINE_BASEPATH / "dev"
MAPPING_PRETRAINED_DIR = xconstants.MAPPING_BASELINE_BASEPATH / "dev"

experiment_list = [
    Exp(name="fer_be_vad",
        parent_dir=PARENTDIR,
        dataset_class=data.images.FER2013Vad,
        image_checkpoint=get_best_checkpoint(get_experiment_dir(IMAGE_PRETRAINED_DIR, "fer2013_be")),
        intermediate_format="be_fer13",
        mapping_checkpoint=get_best_checkpoint(get_experiment_dir(MAPPING_PRETRAINED_DIR, "FER-BE-VAD")),
        batch_size=128,
        split="dev"),
    Exp(name="fer_vad_be",
        parent_dir=PARENTDIR,
        dataset_class=data.images.FER2013,
        image_checkpoint=get_best_checkpoint(get_experiment_dir(IMAGE_PRETRAINED_DIR, "fer2013vad")),
        intermediate_format="vad",
        mapping_checkpoint=get_best_checkpoint(get_experiment_dir(MAPPING_PRETRAINED_DIR, "FER-VAD-BE")),
        batch_size=128,
        split="dev"),
    Exp(name="affectnet_be_va",
        parent_dir=PARENTDIR,
        dataset_class=data.images.AffectNet2019_VA,
        image_checkpoint=get_best_checkpoint(get_experiment_dir(IMAGE_PRETRAINED_DIR, "affectnet_be")),
        intermediate_format="be_affectnet",
        mapping_checkpoint=get_best_checkpoint(get_experiment_dir(MAPPING_PRETRAINED_DIR, "AffectNet-BE-VA")),
        batch_size=128,
        split="dev"),
    Exp(name="affectnet_va_be",
        parent_dir=PARENTDIR,
        dataset_class=data.images.AffectNet2019_BE,
        image_checkpoint=get_best_checkpoint(get_experiment_dir(IMAGE_PRETRAINED_DIR, "affectnet_va")),
        intermediate_format="va",
        mapping_checkpoint=get_best_checkpoint(get_experiment_dir(MAPPING_PRETRAINED_DIR, "AffectNet-VA-BE")),
        batch_size=128,
        split="dev"),


]

def run_all_dev_exp(gpu):
    for x in experiment_list:
        x.run(gpu=gpu)

def run_all_test_exp(gpu):
    for dev_experiment in experiment_list:
        test_experiment = Exp(name=dev_experiment.base_name,
                              parent_dir=TESTDIR,
                              dataset_class=dev_experiment.dataset_class,
                              intermediate_format=dev_experiment.intermediate_format,
                              image_checkpoint=dev_experiment.image_checkpoint,
                              mapping_checkpoint=dev_experiment.mapping_checkpoint,
                              split="test",
                              batch_size=32)

        test_experiment.run(gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="Whether to run test set experiments rather than dev experiments.",
                        action="store_true")
    parser.add_argument("--gpu", help="Which gpu to run on", default=0, type=int)
    args = parser.parse_args()

    if args.test:
        run_all_test_exp(gpu=args.gpu)
    else:
        run_all_dev_exp(gpu=args.gpu)
