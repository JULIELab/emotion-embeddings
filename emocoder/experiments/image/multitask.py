import logging
from typing import Type,List
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from emocoder.src import utils, data, constants
from emocoder.src.experiments import Experiment, get_best_checkpoint
from  emocoder.experiments.mapping import multitask as multitask_mapping_experiment
from emocoder.experiments.image.proposed import ImageProposed
from emocoder.experiments import utils as xutils, constants as xconstants
from emocoder.experiments.experiment_classes import Checkpoint_Test_Image


class ImageMultitaskExperiment(Experiment):

    def __init__(self,
                 name:str,
                 parent_dir: Path,
                 dataset_classes: List[Type[data.images.ImageDataset]],
                 epochs,
                 train_batchsizes: List[int],
                 test_batchsizes: List[int],
                 batches_per_epoch: int):
        super().__init__(name=name, parent_dir=parent_dir, greater_is_better=True, performance_key="overall_mean")

        # self.decoder = decoder
        self.dataset_classes = dataset_classes
        self.epochs = epochs
        self.train_batchsizes = train_batchsizes
        self.test_batchsizes = test_batchsizes
        self.batches_per_epoch = batches_per_epoch


    @staticmethod
    def get_model():
        model, state_dict = ImageProposed.get_model()
        return model, state_dict



    def run(self, gpu=0):
        self.setup()

        logging.info(f"Image Proposed experiment run on {self.dataset_classes} using pretrained ResNet.")

        logging.info("Building model...")

        model, orginal_state_dict = self.get_model()


        logging.info(f"{model}")

        logging.info(f"Setting up gpu {gpu}...")
        gpu = torch.device(f"cuda:{gpu}")
        model.to(device=gpu)

        logging.info("Loading data...")
        train_transform = data.images.get_ResNet_Preprocessor(data_augmentation=True)
        test_transform = data.images.get_ResNet_Preprocessor(data_augmentation=False)

        dataloaders = []
        for i,dsc in enumerate(self.dataset_classes):
            ds = dsc("train", transform=train_transform)
            dl = DataLoader(dataset=ds, batch_size=self.train_batchsizes[i], shuffle=True, num_workers=4)
            dataloaders.append(dl)
        dl = data.utils.MultiDataLoaderIterator2(dataloaders, self.batches_per_epoch) # TODO Delete MultiDataLoaderIterator (1). I don't think I'm using it.


        logging.info("Preparing training...")
        optimizer = torch.optim.SGD(model.enc.parameters(), lr=0.001, momentum=0.9)
        logging.info(optimizer)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=.5)
        logging.info(scheduler)
        loss_fns = {dsc.format: dsc.loss(reduction="mean") for dsc in self.dataset_classes}
        logging.info(loss_fns)

        logging.info("Starting to train...")
        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}. Current learning rate is {scheduler.get_lr()}.")
            model.train()
            epoch_loss = torch.tensor(0.)
            for batch in dl:
                optimizer.zero_grad()
                dsc = batch["dataloader"].dataset.__class__
                out_format = dsc.format
                features = batch["features"].to(gpu)
                labels = batch["labels"].to(gpu)
                preds = model(features, encoder="images", decoders=out_format)
                loss = loss_fns[out_format](preds, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            epoch_loss = epoch_loss.item()
            logging.info(f"Train loss in epoch {i_epoch}: {epoch_loss}")


            # validation
            results = {}
            losses = {}
            for i, dsc in enumerate(self.dataset_classes):
                model.set_default(encoder="images", decoders=dsc.format)
                metric, loss = dsc.score(model=model,
                                             device=gpu,
                                             split="dev",
                                             transform=test_transform,
                                             batch_size=self.test_batchsizes[i])
                result = metric.result()
                results[dsc.__name__] = result
                losses[dsc.__name__] = loss

            results[self.performance_key] = np.mean([results[dsc.__name__][dsc.performance_key] for dsc in self.dataset_classes])

            logging.info(f"Perfomance in epoch {i_epoch}: {results}")
            self.save_results(i_epoch, results)
            self.save_tensorboard_scalars("loss", {"train": epoch_loss, "dev": losses}, i_epoch)
            self.save_tensorboard_scalars("devset_performance",result, i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()
        self.end_experiment()





BASEPATH = utils.get_project_root() / "emocoder" / "target" / "image" / "multitask" / "dev"
TESTDIR = utils.get_project_root() / "emocoder" / "target" / "image" / "multitask" / "test"

EXPERIMENTS = {
    "fer": ImageMultitaskExperiment(name="fer_multitask",
                  parent_dir=BASEPATH,
                  dataset_classes=[data.images.FER2013, data.images.FER2013Vad],
                  epochs=42,
                  train_batchsizes=[32, 32],
                  test_batchsizes=[128, 128],
                  batches_per_epoch=1000),
    "affectnet": ImageMultitaskExperiment(name="affectnet_multitask",
                                    parent_dir=BASEPATH,
                                    dataset_classes=[data.images.AffectNet2019_BE, data.images.AffectNet2019_VA],
                                    epochs=20,
                                    train_batchsizes=[32, 32],
                                    test_batchsizes=[128, 128],
                                    batches_per_epoch=10000),
}


def run_all_dev_exp(gpu=0):
    for x in EXPERIMENTS.values():
        x.run(gpu=gpu)


def run_all_test_exp(gpu):
    for exp_name, dev_experiment in EXPERIMENTS.items():

        model, __ = dev_experiment.get_model()
        checkpoint = get_best_checkpoint(utils.get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name))

        for dataset_class in dev_experiment.dataset_classes:
            model.set_default(encoder="images", decoders=dataset_class.format)
            test_experiment = Checkpoint_Test_Image(name=dev_experiment.base_name,
                                                    parent_dir=TESTDIR,
                                                    dataset_class=dataset_class,
                                                    split="test",
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