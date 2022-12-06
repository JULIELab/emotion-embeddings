import argparse

from emocoder.src import data, models, experiments, constants, metrics, utils
from torch.utils.data import DataLoader
import logging
import torch
from typing import Union
from emocoder.experiments import constants as xconstants, utils as xutils, experiment_classes
from  emocoder.experiments.mapping import multitask as multitask_mapping_experiment
from emocoder.src.experiments import get_best_checkpoint
from emocoder.src.utils import get_experiment_dir


class Word_Proposed(experiments.Experiment):
    """
    Load pretrained BE5 and VAD decoders. Fit encoders for specific languages, one for each data set.
    But not joint learning on multiple datasets yet.
    """

    def __init__(self, name:str,
                 parent_dir,
                 dataset_class,
                 epochs,
                 embeddings:Union[str, type(data.vectors.Embedding_Model)],
                 embedding_limit):
        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)
        self.epochs =epochs
        self.dataset_class = dataset_class
        self.embeddings = embeddings
        self.embedding_limit = embedding_limit


    @staticmethod
    def get_model():
        model, state_dict = xutils.get_pretrained_emotion_codec()
        model.enc["words"] = models.WordFFN(num_outputs=model.enc.proj.size, scaling="logits")
        return model, state_dict


    def run(self,gpu=0):
        self.setup()
        logging.info("Folder structure set up")

        logging.info("Building model, loading decoder checkpoint")
        model, original_state_dict = self.get_model()
        gpu = torch.device(f"cuda:{gpu}")
        model.to(device=gpu)
        model.set_default(encoder="words", decoders=self.dataset_class.format)


        logging.info("Preparing data")
        emb_transform = data.utils.Embedding_Lookup_Transform(embeddings=self.embeddings, limit=self.embedding_limit)
        ds = self.dataset_class("train", transform=emb_transform, scale=True)
        dl = DataLoader(dataset=ds, shuffle=True, batch_size=128)

        logging.info("Preparing training")
        optimizer = torch.optim.Adam(params=model.enc.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        model.dec.eval()

        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}.")
            model.enc.train()
            epoch_loss = torch.tensor(0.)
            for batch in dl:
                optimizer.zero_grad()
                features = batch["features"].to(gpu)
                labels = batch["labels"].to(gpu)
                preds = model(features) #default encoder and decoders already set above
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            logging.info(f"Loss in epoch {i_epoch}: {epoch_loss.item()}")
            metric, val_loss = self.dataset_class.score(model, device=gpu, split="dev", transform=emb_transform)
            result = metric.result()
            logging.info(f"Perfomance in epoch {i_epoch}: {result}")

            self.save_results(i_epoch, result)
            self.save_tensorboard_scalars("devset_pearson", result, i_epoch)
            self.save_tensorboard_scalars("loss", epoch_loss.item(), i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()

        self.compare_model_with_state_dict(model, original_state_dict)
        self.end_experiment()


TARGETDIR = xconstants.WORD_PROPOSED_BASEPATH / "dev"
TESTDIR = xconstants.WORD_PROPOSED_BASEPATH / "test"

experiment_list = [
    # Word_Proposed(name="xanewVAD",
    #               parent_dir=TARGETDIR,
    #               dataset_class=data.words.XANEW,
    #               embeddings="FB_CC_EN",
    #               embedding_limit=None,
    #               epochs=70),
    # Word_Proposed(name="xanewBE",
    #               parent_dir=TARGETDIR,
    #               dataset_class=data.words.XANEW_BE,
    #               embeddings="FB_CC_EN",
    #               embedding_limit=None,
    #               epochs=70),
    Word_Proposed(name="anew",
                  parent_dir=TARGETDIR,
                  dataset_class=data.words.ANEW1999,
                  embeddings="FB_CC_EN",
                  embedding_limit=None,
                  epochs=70),
    Word_Proposed(name="stevenson",
                  parent_dir=TARGETDIR,
                  dataset_class=data.words.Stevenson2007,
                  embeddings="FB_CC_EN",
                  embedding_limit=None,
                  epochs=70),
    Word_Proposed(name="stadthagenVA",
                  parent_dir=TARGETDIR,
                  dataset_class=data.words.Stadthagen_VA,
                  embeddings="FB_CC_ES",
                  embedding_limit=None,
                  epochs=70),
    Word_Proposed(name="stadthagenBE",
                  parent_dir=TARGETDIR,
                  dataset_class=data.words.Stadthagen_BE,
                  embeddings="FB_CC_ES",
                  embedding_limit=None,
                  epochs=70),
    Word_Proposed(name="vo",
                  parent_dir=TARGETDIR,
                  dataset_class=data.words.Vo,
                  embeddings=data.vectors.Facebook_CommonCrawl_German,
                  embedding_limit=None,
                  epochs=70),
    Word_Proposed(name="briesemeister",
                  parent_dir=TARGETDIR,
                  dataset_class=data.words.Briesemeister,
                  embeddings=data.vectors.Facebook_CommonCrawl_German,
                  embedding_limit=None,
                  epochs=70),
    Word_Proposed(name="riegel",
                  parent_dir=TARGETDIR,
                  dataset_class=data.words.Riegel,
                  embeddings=data.vectors.Facebook_CommonCrawl_Polish,
                  embedding_limit=None,
                  epochs=70),
    Word_Proposed(name="wierzba",
                  parent_dir=TARGETDIR,
                  dataset_class=data.words.Wierzba,
                  embeddings=data.vectors.Facebook_CommonCrawl_Polish,
                  embedding_limit=None,
                  epochs=70),
    Word_Proposed(name="kapucuVA",
                  parent_dir=TARGETDIR,
                  dataset_class=data.words.Kapucu_VA,
                  embeddings="FB_CC_TR",
                  embedding_limit=None,
                  epochs=70),
    Word_Proposed(name="kapucuBE",
                  parent_dir=TARGETDIR,
                  dataset_class=data.words.Kapucu_BE,
                  embeddings="FB_CC_TR",
                  embedding_limit=None,
                  epochs=70),
]


def run_all_dev_exp(gpu):
    for x in experiment_list:
        x.run(gpu)

def run_all_test_exp(gpu):
    for dev_experiment in experiment_list:

        model, __  = dev_experiment.get_model()
        model.set_default(encoder="words", decoders=dev_experiment.dataset_class.format)
        checkpoint = get_best_checkpoint(get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name))

        test_experiment = experiment_classes.Checkpoint_Test_Word(name=dev_experiment.base_name,
                                                                  parent_dir=TESTDIR,
                                                                  dataset_class=dev_experiment.dataset_class,
                                                                  split="test",
                                                                  embeddings=dev_experiment.embeddings,
                                                                  embedding_limit=dev_experiment.embedding_limit,
                                                                  model=model,
                                                                  checkpoint=checkpoint)
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