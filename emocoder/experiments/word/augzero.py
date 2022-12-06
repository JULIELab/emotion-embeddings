import argparse

from emocoder.src import data, experiments, utils, constants
from emocoder.experiments.word.proposed import Word_Proposed
from emocoder.experiments.experiment_classes import Checkpoint_Test_Word
from emocoder.experiments import constants as xconstants
from emocoder.src.utils import get_experiment_dir
from emocoder.src.experiments import get_best_checkpoint
from pathlib import Path


PARENTDIR = xconstants.WORD_AUGZERO_BASEPATH / "dev"
TESTDIR = xconstants.WORD_AUGZERO_BASEPATH / "test"
PRETRAINED_DIR = xconstants.WORD_AUGMENTED_BASEPATH / "dev"


EMBEDDING_LIMIT = None



experiment_list = [
    # Checkpoint_Test_Word(name="xanewVAD", parent_dir=PARENTDIR, dataset_class=data.words.XANEW, split="dev",
    #                      embeddings=data.vectors.Facebook_CommonCrawl_English, embedding_limit=EMBEDDING_LIMIT,
    #                      model=Word_Proposed.get_model()[0].select(encoder="words", decoders="vad"),
    #                      checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "XANEW_BE"))),
    # Checkpoint_Test_Word(name="xanewBE", parent_dir=PARENTDIR, dataset_class=data.words.XANEW_BE,
    #                      split="dev", embeddings=data.vectors.Facebook_CommonCrawl_English, embedding_limit=EMBEDDING_LIMIT,
    #                      model=Word_Proposed.get_model()[0].select(encoder="words", decoders="be5"),
    #                      checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "XANEW-Augmented"))),
    Checkpoint_Test_Word(name="anew",
                         parent_dir=PARENTDIR,
                         dataset_class=data.words.ANEW1999,
                         split="dev",
                         embeddings=data.vectors.Facebook_CommonCrawl_English,
                         embedding_limit=EMBEDDING_LIMIT,
                         model=Word_Proposed.get_model()[0].select(encoder="words", decoders="vad"),
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "stevenson"))),
    Checkpoint_Test_Word(name="stevenson",
                         parent_dir=PARENTDIR,
                         dataset_class=data.words.Stevenson2007,
                         split="dev",
                         embeddings=data.vectors.Facebook_CommonCrawl_English,
                         embedding_limit=EMBEDDING_LIMIT,
                         model=Word_Proposed.get_model()[0].select(encoder="words", decoders="be5"),
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "anew"))),
    Checkpoint_Test_Word(name="stadthagenVA", parent_dir=PARENTDIR,
                         dataset_class=data.words.Stadthagen_VA, split="dev",
                         embeddings=data.vectors.Facebook_CommonCrawl_Spanish, embedding_limit=EMBEDDING_LIMIT,
                         model=Word_Proposed.get_model()[0].select(encoder="words", decoders="va"),
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "stadthagenBE"))),
    Checkpoint_Test_Word(name="stadthagenBE", parent_dir=PARENTDIR,
                         dataset_class=data.words.Stadthagen_BE, split="dev",
                         embeddings=data.vectors.Facebook_CommonCrawl_Spanish, embedding_limit=EMBEDDING_LIMIT,
                         model=Word_Proposed.get_model()[0].select(encoder="words", decoders="be5"),
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "stadthagenVA"))),
    Checkpoint_Test_Word(name="vo", parent_dir=PARENTDIR, dataset_class=data.words.Vo, split="dev",
                         embeddings=data.vectors.Facebook_CommonCrawl_German, embedding_limit=EMBEDDING_LIMIT,
                         model=Word_Proposed.get_model()[0].select(encoder="words", decoders="va"),
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "briesemeister"))),
    Checkpoint_Test_Word(name="briesemeister", parent_dir=PARENTDIR,
                         dataset_class=data.words.Briesemeister, split="dev",
                         embeddings=data.vectors.Facebook_CommonCrawl_German, embedding_limit=EMBEDDING_LIMIT,
                         model=Word_Proposed.get_model()[0].select(encoder="words", decoders="be5"),
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "vo"))),
    Checkpoint_Test_Word(name="riegel", parent_dir=PARENTDIR, dataset_class=data.words.Riegel,
                         split="dev", embeddings=data.vectors.Facebook_CommonCrawl_Polish, embedding_limit=EMBEDDING_LIMIT,
                         model=Word_Proposed.get_model()[0].select(encoder="words", decoders="va"),
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "wierzba"))),
    Checkpoint_Test_Word(name="wierzba", parent_dir=PARENTDIR, dataset_class=data.words.Wierzba,
                         split="dev", embeddings=data.vectors.Facebook_CommonCrawl_Polish, embedding_limit=EMBEDDING_LIMIT,
                         model=Word_Proposed.get_model()[0].select(encoder="words", decoders="be5"),
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "riegel"))),
    Checkpoint_Test_Word(name="kapucuVA", parent_dir=PARENTDIR, dataset_class=data.words.Kapucu_VA,
                         split="dev", embeddings=data.vectors.Facebook_CommonCrawl_Turkish, embedding_limit=EMBEDDING_LIMIT,
                         model=Word_Proposed.get_model()[0].select(encoder="words", decoders="va"),
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "kapucuBE"))),
    Checkpoint_Test_Word(name="kapucuBE",
                         parent_dir=PARENTDIR,
                         dataset_class=data.words.Kapucu_BE,
                         split="dev",
                         embeddings=data.vectors.Facebook_CommonCrawl_Turkish,
                         embedding_limit=EMBEDDING_LIMIT,
                         model=Word_Proposed.get_model()[0].select(encoder="words", decoders="be5"),
                         checkpoint=get_best_checkpoint(get_experiment_dir(PRETRAINED_DIR, "kapucuVA"))),
]


def run_all_dev_exp(gpu):
    for exp in experiment_list:
        exp.run(gpu)


def run_all_test_exp(gpu):
    for dev_experiment in experiment_list:

        test_experiment = Checkpoint_Test_Word(name=dev_experiment.base_name,
                                                                  parent_dir=TESTDIR,
                                                                  dataset_class=dev_experiment.dataset_class,
                                                                  split="test",
                                                                  embeddings=dev_experiment.embeddings,
                                                                  embedding_limit=dev_experiment.embedding_limit,
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