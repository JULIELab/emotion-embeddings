import logging
from typing import Type
from pathlib import Path

import argparse
import torch
from torch.utils.data import DataLoader

from emocoder.src import utils, data, models, metrics, constants
from emocoder.src.experiments import Experiment
from emocoder.experiments.word.baseline import WordBaseline as WordBaselineExperiment
from emocoder.experiments.mapping.baseline import MappingBaseline as MappingBaselineExperiment
from emocoder.experiments import constants as xconstants
from emocoder.src.utils import get_experiment_dir
from emocoder.src.experiments import get_best_checkpoint


class Exp(Experiment):

    def __init__(self, name, parent_dir, embedding_str, dataset_class: data.words.WordDataset.__class__, intermediate_format,
                 word_checkpoint: Path, mapping_checkpoint: Path, split, embedding_limit=None):

        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)
        self.embeddings = embedding_str
        #self.epochs = epochs
        self.dataset_class = dataset_class
        self.embedding_limit = embedding_limit
        self.intermediate_format = intermediate_format
        self.word_checkpoint = word_checkpoint
        self.mapping_checkpoint = mapping_checkpoint
        self.split = split


    @staticmethod
    def get_model(num_intermediate, intermediate_scaling, num_output, output_scaling):
        model = torch.nn.Sequential(WordBaselineExperiment.get_model(num_outputs=num_intermediate,
                                                                     scaling=intermediate_scaling),
                                    MappingBaselineExperiment.get_model(num_inputs=num_intermediate,
                                                                        num_outputs=num_output,
                                                                        scaling=output_scaling))
        return model

    def run(self, gpu):
        self.setup()

        logging.info("Loading model...")
        model = self.get_model(num_intermediate=len(constants.VARIABLES[self.intermediate_format]),
                               intermediate_scaling="logits",
                               num_output=len(self.dataset_class.variables),
                               output_scaling="logits")
        gpu = torch.device(f"cuda:{gpu}")
        model.to(device=gpu)
        model[0].load_state_dict(torch.load(self.word_checkpoint))
        model[1].load_state_dict(torch.load(self.mapping_checkpoint))

        logging.info("Loading embeddings...")
        emb_transform = data.utils.Embedding_Lookup_Transform(embeddings=self.embeddings, limit=self.embedding_limit)

        logging.info("Start testing (there is no training phase in this experiment)")
        metric, test_loss = self.dataset_class.score(model, device=gpu, split=self.split, transform=emb_transform)
        result = metric.result()

        logging.info(f"Result obtained in zero-shot scenario: {result}")
        self.save_results("zeroshot_perf", result)
        self.end_experiment()

BASEPATH = xconstants.WORD_ZEROSHOTBASELINE_BASEPATH / "dev"
TESTPATH  = xconstants.WORD_ZEROSHOTBASELINE_BASEPATH/ "test"
WORD_PRETRAINED_DIR = xconstants.WORD_BASELINE_BASEPATH / "dev"
MAPPING_PRETRAINED_DIR = xconstants.MAPPING_BASELINE_BASEPATH / "dev"

EMBEDDING_LIMIT = None

experiment_list = [
    # Exp(name="xanewVA", parent_dir=BASEPATH, embedding_str="FB_CC_EN", dataset_class=data.XANEW,
    #     intermediate_format="be5",
    #     word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "xanew_be")),
    #     mapping_checkpoint=get_best_checkpoint(get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_be5_to_vad")),
    #     split="dev",
    #     embedding_limit=EMBEDDING_LIMIT),
    # Exp(name="xanewBE", parent_dir=BASEPATH, embedding_str="FB_CC_EN", dataset_class=data.XANEW_BE,
    #     intermediate_format="vad",
    #     word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "xanew_baseline")),
    #     mapping_checkpoint=get_best_checkpoint(
    #         get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_vad_to_be5")), split="dev",
    #     embedding_limit=EMBEDDING_LIMIT),
    Exp(name="anew",
        parent_dir=BASEPATH,
        embedding_str="FB_CC_EN",
        dataset_class=data.words.ANEW1999,
        intermediate_format="be5",
        word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "stevenson")),
        mapping_checkpoint=get_best_checkpoint(
            get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_be5_to_vad")),
        split="dev",
        embedding_limit=EMBEDDING_LIMIT),
    Exp(name="stevenson",
        parent_dir=BASEPATH,
        embedding_str="FB_CC_EN",
        dataset_class=data.words.Stevenson2007,
        intermediate_format="vad",
        word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "anew")),
        mapping_checkpoint=get_best_checkpoint(
            get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_vad_to_be5")),
        split="dev",
        embedding_limit=EMBEDDING_LIMIT),
    Exp(name="stadthagenVA",
        parent_dir=BASEPATH,
        embedding_str="FB_CC_ES",
        dataset_class=data.words.Stadthagen_VA,
        intermediate_format="be5",
        word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "stadthagenBE")),
        mapping_checkpoint=get_best_checkpoint(get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_be5_to_va-")),
        split="dev",
        embedding_limit=EMBEDDING_LIMIT),
    Exp(name="stadthagenBE",
        parent_dir=BASEPATH,
        embedding_str="FB_CC_ES",
        dataset_class=data.words.Stadthagen_BE,
        intermediate_format="va",
        word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "stadthagenVA")),
        mapping_checkpoint=get_best_checkpoint(
            get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_va_to_be5")),
        split="dev",
        embedding_limit=EMBEDDING_LIMIT),
    Exp(name="vo",
        parent_dir=BASEPATH,
        embedding_str="FB_CC_DE",
        dataset_class=data.words.Vo,
        intermediate_format="be5",
        word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "briesemeister")),
        mapping_checkpoint=get_best_checkpoint(
            get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_be5_to_va-")),
        split="dev",
        embedding_limit=EMBEDDING_LIMIT),
    Exp(name="briesemeister",
        parent_dir=BASEPATH,
        embedding_str="FB_CC_DE",
        dataset_class=data.words.Briesemeister,
        intermediate_format="va",
        word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "vo")),
        mapping_checkpoint=get_best_checkpoint(
            get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_va_to_be5")),
        split="dev",
        embedding_limit=EMBEDDING_LIMIT),
    Exp(name="riegel",
        parent_dir=BASEPATH,
        embedding_str="FB_CC_PL",
        dataset_class=data.words.Riegel,
        intermediate_format="be5",
        word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "wierzba")),
        mapping_checkpoint=get_best_checkpoint(
            get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_be5_to_va-")),
        split="dev",
        embedding_limit=EMBEDDING_LIMIT),
    Exp(name="wierzba",
        parent_dir=BASEPATH,
        embedding_str="FB_CC_PL",
        dataset_class=data.words.Wierzba,
        intermediate_format="va",
        word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "riegel")),
        mapping_checkpoint=get_best_checkpoint(
            get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_va_to_be5")),
        split="dev",
        embedding_limit=EMBEDDING_LIMIT),
    Exp(name="kapucuVA",
        parent_dir=BASEPATH,
        embedding_str="FB_CC_TR",
        dataset_class=data.words.Kapucu_VA,
        intermediate_format="be5",
        word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "kapucuBE")),
        mapping_checkpoint=get_best_checkpoint(
            get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_be5_to_va-")),
        split="dev",
        embedding_limit=EMBEDDING_LIMIT),
    Exp(name="kapucuBE",
        parent_dir=BASEPATH,
        embedding_str="FB_CC_TR",
        dataset_class=data.words.Kapucu_BE,
        intermediate_format="va",
        word_checkpoint=get_best_checkpoint(get_experiment_dir(WORD_PRETRAINED_DIR, "kapucuVA")),
        mapping_checkpoint=get_best_checkpoint(
            get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_va_to_be5")),
        split="dev",
        embedding_limit=EMBEDDING_LIMIT)
]


def run_all_dev_exp(gpu):
    for x in experiment_list:
        x.run(gpu)

def run_all_test_exp(gpu):
    for dev_experiment in experiment_list:

        test_experiment = Exp(name=dev_experiment.base_name,
                              parent_dir= TESTPATH,
                              dataset_class=dev_experiment.dataset_class,
                              intermediate_format=dev_experiment.intermediate_format,
                              word_checkpoint=dev_experiment.word_checkpoint,
                              mapping_checkpoint=dev_experiment.mapping_checkpoint,
                              split="test",
                              embedding_str=dev_experiment.embeddings,
                              embedding_limit=dev_experiment.embedding_limit)
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