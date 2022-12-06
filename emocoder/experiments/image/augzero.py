from emocoder.src import data, experiments, utils
import logging
import argparse
import torch
from typing import Union
from emocoder.experiments.experiment_classes import Checkpoint_Test_Image
from emocoder.experiments.image.proposed import ImageProposed
from emocoder.src.experiments import get_best_checkpoint
from emocoder.src.utils import get_experiment_dir
import emocoder.experiments.constants as xconstants

PARENTDIR = xconstants.IMAGE_AUGZERO_BASEPATH / "dev"
TESTDIR = xconstants.IMAGE_AUGZERO_BASEPATH / "test"
PRETRAINED_DIR = xconstants.IMAGE_AUGMENTED_BASEPATH / "dev"

EXPERIMENTS = {
    "fer_be_va": Checkpoint_Test_Image(name="fer_be_va",
                                       parent_dir=PARENTDIR,
                                       dataset_class=data.images.FER2013Vad,
                                       split="dev",
                                       model=ImageProposed.get_model()[0].select(encoder="images", decoders="vad"),
                                       checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "fer_be"))),
    "fer_va_be": Checkpoint_Test_Image(name="fer_va_be",
                                       parent_dir=PARENTDIR,
                                       dataset_class=data.images.FER2013,
                                       split="dev",
                                       model=ImageProposed.get_model()[0].select(encoder="images", decoders="be_fer13"),
                                       checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "fer_va"))),
    "affectnet_be_va": Checkpoint_Test_Image(name="affectnet_be_va",
                                             parent_dir=PARENTDIR,
                                             dataset_class=data.images.AffectNet2019_VA,
                                             split="dev",
                                             model=ImageProposed.get_model()[0].select(encoder="images", decoders="va"),
                                             checkpoint=get_best_checkpoint(
                                                 get_experiment_dir(PRETRAINED_DIR, "affectnet_be"))),
    "affectnet_va_be": Checkpoint_Test_Image(name="affectnet_va_be",
                                             parent_dir=PARENTDIR,
                                             dataset_class=data.images.AffectNet2019_BE,
                                             split="dev",
                                             model=ImageProposed.get_model()[0].select(encoder="images", decoders="be_affectnet"),
                                             checkpoint=get_best_checkpoint(
                                                 get_experiment_dir(PRETRAINED_DIR, "affectnet_va")))
}


def run_all_dev_exp(gpu):
    for xname, x in EXPERIMENTS.items():
        x.run(gpu=gpu)


def run_all_test_exp(gpu):
    for exp_name, dev_experiment in EXPERIMENTS.items():
        test_experiment = Checkpoint_Test_Image(name=dev_experiment.base_name,
                                                parent_dir=TESTDIR,
                                                dataset_class=dev_experiment.dataset_class,
                                                split="test",
                                                model=dev_experiment.model,
                                                checkpoint=dev_experiment.checkpoint)
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
