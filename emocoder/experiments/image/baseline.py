import logging
from typing import Type

import torch
from torch import nn
from torch.utils.data import DataLoader
from emocoder.src import utils, data, models, metrics, constants
from emocoder.src.experiments import Experiment, get_best_checkpoint
import argparse
from emocoder.experiments.experiment_classes import Checkpoint_Test_Image


class ImageBaseline(Experiment):

    def __init__(self,
                 name: str,
                 parent_dir,
                 # performance_key,
                 # greater_is_better,
                 dataset_class,
                 epochs,
                 train_batchsize,
                 test_batchsize):

        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better
                         )

        self.dataset_class = dataset_class
        self.epochs = epochs
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize

    @staticmethod
    def get_model(n_outputs):
        model = models.get_resnet(34, pretrained=True)
        features_num = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(.5),
            torch.nn.Linear(features_num, n_outputs)
        )
        return model


    def run(self, gpu=0):
        self.setup()

        logging.info(f"Image baseline experiment run on {self.dataset_class} using pretrained ResNet.")

        logging.info("Building model...")
        model = self.get_model(n_outputs=len(self.dataset_class.variables))
        logging.info(f"{model}")

        logging.info("Setting up gpu...")
        gpu = torch.device(f"cuda:{gpu}")
        model.to(gpu)

        logging.info("Loading data...")
        train_transform = data.images.get_ResNet_Preprocessor(data_augmentation=True)
        test_transform = data.images.get_ResNet_Preprocessor(data_augmentation=False)
        ds = self.dataset_class(split="train", transform=train_transform)
        dl = DataLoader(dataset=ds, batch_size=self.train_batchsize, shuffle=True, num_workers=16)

        logging.info("Preparing training...")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        logging.info(optimizer)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=.5)
        logging.info(scheduler)
        loss_fn = self.dataset_class.loss(reduction="mean")
        logging.info(loss_fn)

        logging.info("Starting to train...")
        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}. Current learning rate is {scheduler.get_lr()}.")
            model.train()
            epoch_loss = torch.tensor(0.)
            for batch in dl:
                optimizer.zero_grad()
                features = batch["features"].to(gpu)
                labels = batch["labels"].to(gpu)
                preds = model(features)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss
            scheduler.step()
            epoch_loss = epoch_loss.item() / len(ds)
            logging.info(f"Train loss in epoch {i_epoch}: {epoch_loss}")

            # dev set validation
            metric, dev_loss = self.dataset_class.score(model=model,
                                                        device=gpu,
                                                        split="dev",
                                                        transform=test_transform,
                                                        batch_size=self.test_batchsize)
            if not self.performance_order_defined:
                self.order_performance_according_to(metric)

            logging.info(f"Dev loss in epoch {i_epoch}: {dev_loss}")
            result = metric.result()
            self.save_tensorboard_scalars("devset_performance", result, i_epoch)

            logging.info(f"Perfomance in epoch {i_epoch}: {result}")
            self.save_results(i_epoch, result)
            self.save_tensorboard_scalars("loss", {"train": epoch_loss, "dev": dev_loss}, i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()
        self.end_experiment()




BASEPATH = utils.get_project_root() / "emocoder" / "target" /  "image" / "baseline" / "dev"
TESTDIR = utils.get_project_root() / "emocoder" / "target" /  "image" / "baseline" / "test"


EXPERIMENTS = {
    # "iaps": ImageBaseline(name="iaps_baseline",
    #               parent_dir=BASEPATH,
    #               dataset_class=data.images.IAPS2008,
    #               epochs=42,
    #               train_batchsize=32,
    #               test_batchsize=128),
    # "flickr": ImageBaseline(name="you2017_baseline",
    #               parent_dir=BASEPATH,
    #               dataset_class=data.images.YOU2017,
    #               epochs=42,
    #               train_batchsize=32,
    #               test_batchsize=128),
    "fer_be": ImageBaseline(name="fer2013_be_baseline",
                  parent_dir=BASEPATH,
                  dataset_class=data.images.FER2013,
                  epochs=42,
                  train_batchsize=32,
                  test_batchsize=128),
    "fer_vad": ImageBaseline(name="fer2013vad_baseline",
                  parent_dir=BASEPATH,
                  dataset_class=data.images.FER2013Vad,
                  epochs=42,
                  train_batchsize=32,
                  test_batchsize=128),
    "affectnet_be": ImageBaseline(name="affectnet_be_baseline",
                             parent_dir=BASEPATH,
                             dataset_class=data.images.AffectNet2019_BE,
                             epochs=20,
                             train_batchsize=32,
                             test_batchsize=128),
    "affectnet_va": ImageBaseline(name="affectnet_va_baseline",
                                  parent_dir=BASEPATH,
                                  dataset_class=data.images.AffectNet2019_VA,
                                  epochs=20,
                                  train_batchsize=32,
                                  test_batchsize=128),
}


def run_all_dev_exp(gpu=0):
    for x in EXPERIMENTS.values():
        x.run(gpu=gpu)


def run_all_test_exp(gpu):
    for exp_name, dev_experiment in EXPERIMENTS.items():
        checkpoint = get_best_checkpoint(utils.get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name))
        model = dev_experiment.get_model(len(dev_experiment.dataset_class.variables))

        test_experiment = Checkpoint_Test_Image(name=dev_experiment.base_name,
                                                parent_dir=TESTDIR,
                                                dataset_class=dev_experiment.dataset_class,
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
