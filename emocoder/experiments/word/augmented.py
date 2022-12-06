import argparse

from emocoder.src import data, models, experiments, constants, utils
from torch.utils.data import DataLoader
import logging
import torch
from typing import Union
from emocoder.experiments.word.proposed import Word_Proposed
from emocoder.experiments import constants as xconstants
from emocoder.src.experiments import get_best_checkpoint
from emocoder.experiments import experiment_classes
from emocoder.src.utils import get_experiment_dir


class Fit_Word_Encoders_Augmented(experiments.Experiment):
    """Load pretrained decoders for VAD and BE5. Fit word encoders for a particular dataset. But also map
    labels into the other format thus producing additional supervision.
    """

    def __init__(self,
                 name:str,
                 parent_dir,
                 dataset_class,
                 augmentation_format:str,
                 epochs,
                 embeddings: Union[str, type(data.vectors.Embedding_Model)],
                 embedding_limit):

        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)
        self.epochs =epochs
        self.dataset_class = dataset_class
        self.augmentation_format = augmentation_format
        self.embeddings = embeddings
        self.embedding_limit = embedding_limit


    @staticmethod
    def get_model():
        model, statedict = Word_Proposed.get_model()
        return model, statedict


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
        optimizer = torch.optim.Adam(params=model.enc["words"].parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss(reduction="mean")
        model.eval()

        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}.")
            model.enc["words"].train()
            epoch_loss = torch.tensor(0.)
            for batch in dl:
                optimizer.zero_grad()

                features = batch["features"].to(gpu)
                labels = batch["labels"].to(gpu)
                augmented_labels = model(x=labels, encoder=self.dataset_class.format, decoders=self.augmentation_format)

                embs = model.enc(features, encoder="words")
                preds = model.dec(embs, decoders=self.dataset_class.format)
                augmented_preds = model.dec(embs, decoders=self.augmentation_format)

                loss = loss_fn(preds, labels) + loss_fn(augmented_preds, augmented_labels)

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

PARENTDIR = xconstants.WORD_AUGMENTED_BASEPATH / "dev"
TESTDIR = xconstants.WORD_AUGMENTED_BASEPATH / "test"

experiment_list = [
    # Fit_Word_Encoders_Augmented(name="XANEW-Augmented",
    #                                 parent_dir=PARENTDIR,
    #                                 dataset_class=data.words.XANEW,
    #                                 augmentation_format = "be5",
    #                                 embeddings="FB_CC_EN",
    #                                 embedding_limit=None,
    #                                 epochs=70),
    # Fit_Word_Encoders_Augmented(name="XANEW_BE-Augmented",
    #                                 parent_dir=PARENTDIR,
    #                                 dataset_class=data.words.XANEW_BE,
    #                                 augmentation_format = "vad",
    #                                 embeddings="FB_CC_EN",
    #                                 embedding_limit=None,
    #                                 epochs=70),
    Fit_Word_Encoders_Augmented(name="anew",
                                parent_dir=PARENTDIR,
                                dataset_class=data.words.ANEW1999,
                                augmentation_format="be5",
                                embeddings="FB_CC_EN",
                                embedding_limit=None,
                                epochs=70),
    Fit_Word_Encoders_Augmented(name="stevenson",
                                parent_dir=PARENTDIR,
                                dataset_class=data.words.Stevenson2007,
                                augmentation_format="vad",
                                embeddings="FB_CC_EN",
                                embedding_limit=None,
                                epochs=70),
    Fit_Word_Encoders_Augmented(name="stadthagenVA",
                                    parent_dir=PARENTDIR,
                                    dataset_class=data.words.Stadthagen_VA,
                                    augmentation_format = "be5",
                                    embeddings="FB_CC_ES",
                                    embedding_limit=None,
                                    epochs=70),
    Fit_Word_Encoders_Augmented(name="stadthagenBE",
                                    parent_dir=PARENTDIR,
                                    dataset_class=data.words.Stadthagen_BE,
                                    augmentation_format = "vad",
                                    embeddings="FB_CC_ES",
                                    embedding_limit=None,
                                    epochs=70),
    Fit_Word_Encoders_Augmented(name="vo",
                                    parent_dir=PARENTDIR,
                                    dataset_class=data.words.Vo,
                                    augmentation_format="be5",
                                    embeddings=data.vectors.Facebook_CommonCrawl_German,
                                    embedding_limit=None,
                                    epochs=70),
    Fit_Word_Encoders_Augmented(name="briesemeister",
                                    parent_dir=PARENTDIR,
                                    dataset_class=data.words.Briesemeister,
                                    augmentation_format="vad",
                                    embeddings=data.vectors.Facebook_CommonCrawl_German,
                                    embedding_limit=None,
                                    epochs=70),
    Fit_Word_Encoders_Augmented(name="riegel",
                                    parent_dir=PARENTDIR,
                                    dataset_class=data.words.Riegel,
                                    augmentation_format="be5",
                                    embeddings=data.vectors.Facebook_CommonCrawl_Polish,
                                    embedding_limit=None,
                                    epochs=70),
    Fit_Word_Encoders_Augmented(name="wierzba",
                                    parent_dir=PARENTDIR,
                                    dataset_class=data.words.Wierzba,
                                    augmentation_format="vad",
                                    embeddings=data.vectors.Facebook_CommonCrawl_Polish,
                                    embedding_limit=None,
                                    epochs=70),
    Fit_Word_Encoders_Augmented(name="kapucuVA",
                                    parent_dir=PARENTDIR,
                                    dataset_class=data.words.Kapucu_VA,
                                    augmentation_format="be5",
                                    embeddings="FB_CC_TR",
                                    embedding_limit=None,
                                    epochs=70),
    Fit_Word_Encoders_Augmented(name="kapucuBE",
                                    parent_dir=PARENTDIR,
                                    dataset_class=data.words.Kapucu_BE,
                                    augmentation_format="vad",
                                    embeddings="FB_CC_TR",
                                    embedding_limit=None,
                                    epochs=70),
]


def run_all_dev_exp(gpu):
    for x in experiment_list:
        x.run(gpu)

def run_all_test_exp(gpu):
    for dev_experiment in experiment_list:

        model, __ = dev_experiment.get_model()
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