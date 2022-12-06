from emocoder.src import data
from emocoder.experiments.experiment_classes import Checkpoint_Test_Text
from emocoder.src.utils import get_experiment_dir
from emocoder.src.experiments import get_best_checkpoint
from emocoder.experiments import constants as xconstants
from emocoder.experiments.text.augmented import Exp
from copy import deepcopy
import argparse

PARENTDIR = xconstants.TEXT_AUGZERO_BASEPATH / "dev"
TESTDIR = xconstants.TEXT_AUGZERO_BASEPATH / "test"
PRETRAINED_DIR = xconstants.TEXT_AUGMENTED_BASEPATH / "dev"


en_model, en_tokenizer, __ = Exp.get_model("bert-base-uncased")


experiment_list = [

    Checkpoint_Test_Text(name="emobank",
                         parent_dir=PARENTDIR,
                         dataset_class=data.text.EmoBank,
                         split="dev",
                         model=deepcopy(en_model).select(encoder="text", decoders="vad"),
                         tokenizer=en_tokenizer,
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "afftext")),
                         test_batchsize=48),

    Checkpoint_Test_Text(name="afftext",
                         parent_dir=PARENTDIR,
                         dataset_class=data.text.AffTextBE5,
                         split="dev",
                         model=deepcopy(en_model).select(encoder="text", decoders="be5"),
                         tokenizer=en_tokenizer,
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "emobank")),
                         test_batchsize=48),

    # Checkpoint_Test_Text(name="sst2cls",
    #                      parent_dir=PARENTDIR,
    #                      dataset_class=data.text.SST_2_Class,
    #                      split="dev",
    #                      model=deepcopy(en_model).select(encoder="text", decoders="pol1"),
    #                      tokenizer=en_tokenizer,
    #                      checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "emobank")),
    #                      test_batchsize=48,
    #                      checkpoint_exact_match=False),
]



def run_all_dev_exp(gpu):
    for x in experiment_list:
        x.run(gpu)

def run_all_test_exp(gpu):
    for dev_experiment in experiment_list:

        test_experiment = Checkpoint_Test_Text(name=dev_experiment.base_name,
                                               parent_dir=TESTDIR,
                                               dataset_class=dev_experiment.dataset_class,
                                               split="test",
                                               model=dev_experiment.model,
                                               tokenizer=dev_experiment.tokenizer,
                                               checkpoint=dev_experiment.checkpoint,
                                               test_batchsize=dev_experiment.test_batchsize,
                                               checkpoint_exact_match=dev_experiment.checkpoint_exact_match)
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

