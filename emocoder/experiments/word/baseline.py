import logging
from typing import Type
import argparse

import torch
from torch.utils.data import DataLoader

from emocoder.src import utils, data, models, metrics
from emocoder.src.experiments import Experiment, get_best_checkpoint
from emocoder.experiments import constants as xconstants, experiment_classes
from emocoder.src.utils import get_experiment_dir


class WordBaseline(Experiment):

    def __init__(self,
                 name,
                 parent_dir,
                 embedding_str,
                 epochs,
                 dataset_class:data.words.WordDataset.__class__,
                 embedding_limit=None):

        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)
        self.embeddings = embedding_str
        self.epochs = epochs
        self.dataset_class = dataset_class
        self.embedding_limit = embedding_limit


    @staticmethod
    def get_model(num_outputs, scaling):
        return models.WordFFN(num_outputs=num_outputs, scaling=scaling)



    def run(self, gpu):
        self.setup()

        model: torch.nn.Module = self.get_model(len(self.dataset_class.variables), scaling="logits")
        gpu = torch.device(f"cuda:{gpu}")
        model.to(gpu)

        logging.info("loading embeddings")
        emb_transform = data.utils.Embedding_Lookup_Transform(embeddings=self.embeddings, limit=self.embedding_limit)
        logging.info("loading data")
        train_dl = DataLoader(dataset=self.dataset_class("train", transform=emb_transform),
                              shuffle=True,
                              batch_size=128)

        logging.info("preparing training")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

        loss_fn = self.dataset_class.loss(reduction="mean")

        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}")
            model.train()
            epoch_loss = torch.tensor(0.)
            for batch in train_dl:
                optimizer.zero_grad()
                features = batch["features"].to(gpu)
                labels = batch["labels"].to(gpu)
                preds = model(features)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            epoch_loss = epoch_loss.item()
            logging.info(f"Loss in epoch {i_epoch}: {epoch_loss}")
            metric, dev_loss = self.dataset_class.score(model, device=gpu, split="dev", transform=emb_transform)

            if not self.performance_order_defined:
                self.order_performance_according_to(metric)

            result = metric.result()
            logging.info(f"Perfomance in epoch {i_epoch}: {result}")
            self.save_results(i_epoch, result)
            self.save_tensorboard_scalars("devset_pearson", result, i_epoch)
            self.save_tensorboard_scalars("loss", epoch_loss, i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()
        self.end_experiment()


BASEPATH = xconstants.WORD_BASELINE_BASEPATH / "dev"
TESTPATH = xconstants.WORD_BASELINE_BASEPATH / "test"
EMBEDDING_LIMIT = None
EPOCHS = 70

experiments = {
                "anew1999":  WordBaseline(name="anew",
                                                    parent_dir=BASEPATH,
                                                    embedding_str="FB_CC_EN",
                                                    epochs=EPOCHS,
                                                    dataset_class=data.words.ANEW1999,
                                                        embedding_limit=EMBEDDING_LIMIT),
                "stevenson2007": WordBaseline(name="stevenson",
                                                    parent_dir=BASEPATH,
                                                    embedding_str="FB_CC_EN",
                                                    epochs=EPOCHS,
                                                    dataset_class=data.words.Stevenson2007,
                                                            embedding_limit=EMBEDDING_LIMIT),
                # "xanew": WordBaseline(name="xanewVAD",
                #                                     parent_dir=BASEPATH,
                #                                     embedding_str="FB_CC_EN",
                #                                     epochs=EPOCHS,
                #                                     dataset_class=data.words.XANEW,
                #                                     embedding_limit=EMBEDDING_LIMIT),
                # "nrc": WordBaseline(name="nrc_baseline",
                #                                     parent_dir=BASEPATH,
                #                                     embedding_str="FB_CC_EN",
                #                                     epochs=EPOCHS,
                #                                     dataset_class=data.words.NRC,
                #                                     embedding_limit=EMBEDDING_LIMIT),
                # "nrc8": WordBaseline(name="nrc8_baseline",
                #                                                parent_dir=BASEPATH,
                #                                                embedding_str="FB_CC_EN",
                #                                                epochs=EPOCHS,
                #                                                dataset_class=data.NRC8,
                #                                                embedding_limit=EMBEDDING_LIMIT),
                "stadthagen_va": WordBaseline(name="stadthagenVA",
                                                    parent_dir=BASEPATH,
                                                    embedding_str="FB_CC_ES",
                                                    epochs=EPOCHS,
                                                    dataset_class=data.words.Stadthagen_VA,
                                                                embedding_limit=EMBEDDING_LIMIT),
                "stadthagen_be": WordBaseline(name="stadthagenBE",
                                                    parent_dir=BASEPATH,
                                                    embedding_str="FB_CC_ES",
                                                    epochs=EPOCHS,
                                                    dataset_class=data.words.Stadthagen_BE,
                                                                embedding_limit=EMBEDDING_LIMIT),
                # "moors": WordBaseline(name="moors2013_baseline",
                #                                     parent_dir=BASEPATH,
                #                                     embedding_str="FB_CC_NL",
                #                                     epochs=EPOCHS,
                #                                     dataset_class=data.words.Moors2013,
                #                                     embedding_limit=EMBEDDING_LIMIT),
                "vo": WordBaseline(name="vo",
                                                    parent_dir=BASEPATH,
                                                    embedding_str="FB_CC_DE",
                                                    epochs=EPOCHS,
                                                    dataset_class=data.words.Vo,
                                                    embedding_limit=EMBEDDING_LIMIT),
                "briesemeister": WordBaseline(name="briesemeister",
                                                             parent_dir=BASEPATH,
                                                             embedding_str="FB_CC_DE",
                                                             epochs=EPOCHS,
                                                             dataset_class=data.words.Briesemeister,
                                                                embedding_limit=EMBEDDING_LIMIT),
                # "imbir": WordBaseline(name="imbir_baseline",
                #                                     parent_dir=BASEPATH,
                #                                     embedding_str="FB_CC_PL",
                #                                     epochs=EPOCHS,
                #                                     dataset_class=data.words.Imbir,
                #                                     embedding_limit=EMBEDDING_LIMIT),
                "riegel": WordBaseline(name="riegel",
                                                                parent_dir=BASEPATH,
                                                                embedding_str="FB_CC_PL",
                                                                epochs=EPOCHS,
                                                                dataset_class=data.words.Riegel,
                                                        embedding_limit=EMBEDDING_LIMIT),
                "wierzba": WordBaseline(name="wierzba",
                                                                parent_dir=BASEPATH,
                                                                embedding_str="FB_CC_PL",
                                                                epochs=EPOCHS,
                                                                dataset_class=data.words.Wierzba,
                                                            embedding_limit=EMBEDDING_LIMIT),
                "kapucu_va": WordBaseline(name="kapucuVA",
                                                    parent_dir=BASEPATH,
                                                    embedding_str="FB_CC_TR",
                                                    epochs=EPOCHS,
                                                    dataset_class=data.words.Kapucu_VA,
                                                    embedding_limit=EMBEDDING_LIMIT),
                "kapucu_be": WordBaseline(name="kapucuBE",
                                                    parent_dir=BASEPATH,
                                                    embedding_str="FB_CC_TR",
                                                    epochs=EPOCHS,
                                                    dataset_class=data.words.Kapucu_BE,
                                                    embedding_limit=EMBEDDING_LIMIT)

}


def run_all_dev_exp(gpu):
    for x in experiments.values():
        x.run(gpu)

def run_all_test_exp(gpu):
    for dev_experiment in experiments.values():

        model = dev_experiment.get_model(len(dev_experiment.dataset_class.variables), scaling="logits")
        checkpoint = get_best_checkpoint(get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name))

        test_experiment = experiment_classes.Checkpoint_Test_Word(name=dev_experiment.base_name,
                                                                  parent_dir=TESTPATH,
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
