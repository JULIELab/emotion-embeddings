import logging
from typing import Type

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from emocoder.src import utils, data, models, metrics, constants
from emocoder.src.experiments import Experiment, get_best_checkpoint

from emocoder.experiments import constants as xconstants, experiment_classes
from emocoder.src.utils import get_experiment_dir


class TextBaseline(Experiment):

    def __init__(self,
                 name: str,
                 parent_dir,
                 pretrained_weights: str, #hugging faces model names; 'bert-base-uncased' for English, bert-base-chinese for Chinese
                 epochs,
                 dataset_class: Type[data.text.TextDataset],
                 train_batchsize=24,
                 test_batchsize=48
                 ):
        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)
        self.pretrained_weights = pretrained_weights
        self.epochs = epochs
        self.dataset_class = dataset_class
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize

    @staticmethod
    def get_model(pretrained_weights, num_outputs):
        model, tokenizer = models.get_transformer_sequence_regressor(pretrained_weights=pretrained_weights,
                                                                     num_outputs=num_outputs)
        return model,tokenizer

    def run(self, gpu):
        self.setup()

        logging.info(f"Text experiment run on {self.dataset_class} using pretrained weighs {self.pretrained_weights}.")

        logging.info("Building model")
        model, tokenizer = self.get_model(pretrained_weights=self.pretrained_weights,
                                          num_outputs=len(self.dataset_class.variables))

        logging.info("setting up gpu")
        gpu = torch.device(f"cuda:{gpu}")
        model.to(gpu)

        logging.info("loading data")
        transform, collater = data.utils.get_text_transform_and_collater(dataset_class=self.dataset_class,
                                                                         tokenizer=tokenizer)

        dataset = self.dataset_class("train", transform=transform)
        train_dl = DataLoader(dataset=dataset, batch_size=self.train_batchsize, shuffle=True, num_workers=16, collate_fn=collater)


        logging.info("preparing training")
        optimizer = AdamW(model.parameters(), lr=1e-5)
        loss_fn = self.dataset_class.loss()


        logging.info("start training")
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

            # dev set eval
            metric, __ = self.dataset_class.score(model, device=gpu, split="dev", batch_size=self.test_batchsize, collater=collater,
                                              transform=transform)

            result = metric.result()
            logging.info(f"Perfomance in epoch {i_epoch}: {result}")
            self.save_results(i_epoch, result)
            self.save_tensorboard_scalars("devset_performance", result, i_epoch)
            self.save_tensorboard_scalars("train_loss", epoch_loss, i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()
        self.end_experiment()




BASEPATH = xconstants.TEXT_BASELINE_BASEPATH / "dev"
TESTDIR = xconstants.TEXT_BASELINE_BASEPATH / "test"

experiments = {
               "emobank": TextBaseline(name="emobank",
                                       parent_dir=BASEPATH,
                                       pretrained_weights='bert-base-uncased',
                                       epochs=10,
                                       dataset_class=data.text.EmoBank),
               # "sst5cls": TextBaseline(name="sst",
               #                     parent_dir=BASEPATH,
               #                     pretrained_weights='bert-base-uncased',
               #                     epochs=20,
               #                     dataset_class=data.text.SST_5_Class),
               "isear": TextBaseline(name="isear",
                                   parent_dir=BASEPATH,
                                   pretrained_weights='bert-base-uncased',
                                   epochs=20,
                                   dataset_class=data.text.ISEAR,
                                               train_batchsize=12),
               "ssec": TextBaseline(name="ssec",
                                   parent_dir=BASEPATH,
                                   pretrained_weights='bert-base-uncased',
                                   epochs=20,
                                   dataset_class=data.text.SSEC),
               "cvat": TextBaseline(name="cvat",
                                   parent_dir=BASEPATH,
                                   pretrained_weights="bert-base-chinese",
                                   epochs=20,
                                   dataset_class=data.text.CVAT,
                                                      train_batchsize=12),
               # "afftext_be6": TextBaseline(name="afftext_be6",
               #                         parent_dir=BASEPATH,
               #                         pretrained_weights='bert-base-uncased',
               #                         epochs=20,
               #                         dataset_class=data.text.AffText),
               "afftext_be5": TextBaseline(name="afftext",
                                              parent_dir=BASEPATH,
                                              pretrained_weights='bert-base-uncased',
                                              epochs=40,
                                              dataset_class=data.text.AffTextBE5),
               # "sst2cls": TextBaseline(name="sst2cls",
               #                     parent_dir=BASEPATH,
               #                     pretrained_weights='bert-base-uncased',
               #                     epochs=10,
               #                     dataset_class=data.text.SST_2_Class),
               }


def run_all_dev_exp(gpu):
    global name
    for name, x in experiments.items():
        x.run(gpu)


def run_all_test_exp(gpu):
    for dev_experiment in experiments.values():
        checkpoint = get_best_checkpoint(get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name))

        model, tokenizer = dev_experiment.get_model(pretrained_weights=dev_experiment.pretrained_weights,
                                                    num_outputs=len(dev_experiment.dataset_class.variables))

        test_experiment = experiment_classes.Checkpoint_Test_Text(name=dev_experiment.base_name,
                                                                  parent_dir=TESTDIR,
                                                                  dataset_class=dev_experiment.dataset_class,
                                                                  split="test",
                                                                  model=model,
                                                                  tokenizer=tokenizer,
                                                                  checkpoint=checkpoint,
                                                                  test_batchsize=48)
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


