import logging

import argparse
import torch

from emocoder.experiments import constants as xconstants
from emocoder.experiments.mapping.baseline import MappingBaseline as MappingBaselineExperiment
from emocoder.experiments.text.baseline import TextBaseline as TextBaselineExperiment
from emocoder.src.utils import get_experiment_dir
from emocoder.src import data, constants, utils
from emocoder.src.experiments import Experiment
from emocoder.src.experiments import get_best_checkpoint


class Exp(Experiment):

    def __init__(self,
                 name,
                 parent_dir,
                 dataset_class,
                 pretrained_weights,
                 text_checkpoint,
                 intermediate_format,
                 mapping_checkpoint,
                 batch_size,
                 split,
                 pick_intermediate_output: int = None):
        """
        :param name:
        :param parent_dir:
        :param dataset_class:
        :param pretrained_weights:
        :param text_checkpoint:
        :param intermediate_format:
        :param mapping_checkpoint:
        :param batch_size:
        :param split:
        :param pick_intermediate_output: Int or None. If not None, do not use a an acutal mapping  model on top of the
            text model, but rather just pick one of the elements of the (intermediate) output. This is sort of a hacky
            solution for the special case of inference on SST with EmoBank baseline model.
        """

        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)

        self.dataset_class = dataset_class
        self.pretrained_weights = pretrained_weights
        self.text_checkpoint = text_checkpoint
        self.mapping_checkpoint = mapping_checkpoint
        self.intermediate_format = intermediate_format
        self.batch_size = batch_size
        self.split = split
        self.pick_intermediate_output=pick_intermediate_output


    @staticmethod
    def get_model(pretrained_weights, num_intermediate, num_outputs, output_scaling):
       text_model, tokenizer = TextBaselineExperiment.get_model(pretrained_weights=pretrained_weights,
                                                                num_outputs=num_intermediate)
       mapping_model = MappingBaselineExperiment.get_model(num_inputs=num_intermediate,
                                                           num_outputs=num_outputs,
                                                           scaling=output_scaling)
       model = torch.nn.Sequential(text_model, mapping_model)
       return model, tokenizer


    @staticmethod
    def get_model_pick_intermediate_output(pretrained_weights, num_intermediate, pick):
        text_model, tokenizer = TextBaselineExperiment.get_model(pretrained_weights=pretrained_weights,
                                                                 num_outputs=num_intermediate)

        model = torch.nn.Sequential(text_model, utils.Picking_Layer(pick=pick))
        return model, tokenizer



    def run(self, gpu):
        self.setup()
        gpu = torch.device(f"cuda:{gpu}")

        logging.info("Loading model...")
        if self.pick_intermediate_output is None:
            model, tokenizer = self.get_model(pretrained_weights=self.pretrained_weights,
                                              num_intermediate=len(constants.VARIABLES[self.intermediate_format]),
                                              num_outputs=len(self.dataset_class.variables),
                                              output_scaling="logits")
            model.to(device=gpu)
            model[0].load_state_dict(torch.load(self.text_checkpoint))
            model[1].load_state_dict(torch.load(self.mapping_checkpoint))

        else:
            model, tokenizer = self.get_model_pick_intermediate_output(
                self.pretrained_weights,
                len(constants.VARIABLES[self.intermediate_format]),
                self.pick_intermediate_output)
            model.to(device=gpu)
            model[0].load_state_dict(torch.load(self.text_checkpoint))





        logging.info("loading data")
        transform, collater = data.utils.get_text_transform_and_collater(dataset_class=self.dataset_class,
                                                                         tokenizer=tokenizer)
        logging.info("Start testing (there is no training phase in this experiment)")
        metric, __ = self.dataset_class.score(model, device=gpu, split=self.split, batch_size=self.batch_size, collater=collater,
                                              transform=transform)
        result = metric.result()


        logging.info(f"Result obtained in zero-shot scenario: {result}")
        self.save_results("zeroshot_perf", result)
        self.end_experiment()



BASEPATH = xconstants.TEXT_ZEROSHOTBASELINE_BASEPATH / "dev"
TESTDIR = xconstants.TEXT_ZEROSHOTBASELINE_BASEPATH / "test"
TEXT_PRETRAINED_DIR = xconstants.TEXT_BASELINE_BASEPATH / "dev"
MAPPING_PRETRAINED_DIR = xconstants.MAPPING_BASELINE_BASEPATH / "dev"


experiment_list = [
    Exp(name="emobank",
        parent_dir=BASEPATH,
        dataset_class=data.text.EmoBank,
        pretrained_weights="bert-base-uncased",
        text_checkpoint=get_best_checkpoint(get_experiment_dir(TEXT_PRETRAINED_DIR, "afftext")),
        intermediate_format="be5",
        mapping_checkpoint=get_best_checkpoint(get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_be5_to_vad")),
        batch_size=24,
        split="dev"),

    Exp(name="afftext",
        parent_dir=BASEPATH,
        dataset_class=data.text.AffTextBE5,
        pretrained_weights="bert-base-uncased",
        text_checkpoint=get_best_checkpoint(get_experiment_dir(TEXT_PRETRAINED_DIR, "emobank")),
        intermediate_format="vad",
        mapping_checkpoint=get_best_checkpoint(get_experiment_dir(MAPPING_PRETRAINED_DIR, "anew_stevenson_baseline_vad_to_be5")),
        batch_size=24,
        split="dev"),

    # Exp(name="sst2cls",
    #     parent_dir=BASEPATH,
    #     dataset_class=data.text.SST_2_Class,
    #     pretrained_weights="bert-base-uncased",
    #     text_checkpoint=get_best_checkpoint(get_experiment_dir(TEXT_PRETRAINED_DIR, "emobank")),
    #     intermediate_format="vad",
    #     mapping_checkpoint=None,
    #     batch_size=24,
    #     split="dev",
    #     pick_intermediate_output=0),

]

def run_all_dev_exp(gpu):
    for x in experiment_list:
        x.run(gpu)

def run_all_test_exp(gpu):
    for dev_experiment in experiment_list:

        test_experiment = Exp(name=dev_experiment.base_name,
                              parent_dir=TESTDIR,
                              dataset_class=dev_experiment.dataset_class,
                              pretrained_weights=dev_experiment.pretrained_weights,
                              text_checkpoint=dev_experiment.text_checkpoint,
                              intermediate_format=dev_experiment.intermediate_format,
                              mapping_checkpoint=dev_experiment.mapping_checkpoint,
                              batch_size=dev_experiment.batch_size,
                              split="test",
                              pick_intermediate_output=dev_experiment.pick_intermediate_output)
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




