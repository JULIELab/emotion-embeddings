import logging
from typing import Type
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from emocoder.src import utils, data, constants
from emocoder.src.experiments import Experiment, get_best_checkpoint
from emocoder.experiments.image.baseline import ImageBaseline
from emocoder.experiments import utils as xutils, constants as xconstants
from emocoder.experiments.experiment_classes import Checkpoint_Test_Image
import argparse


class ImageProposed(Experiment):

    def __init__(self,
                 name:str,
                 parent_dir: Path,
                 dataset_class: Type[data.images.ImageDataset],
                 epochs,
                 train_batchsize,
                 test_batchsize,
                 initial_lr=.001,
                 scheduler_step_size=7,
                 scheduler_gamma=.5):
        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)

        # self.decoder = decoder
        self.dataset_class = dataset_class
        self.epochs = epochs
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize
        self.initial_lr =initial_lr
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma


    @staticmethod
    def get_model():
        model, state_dict = xutils.get_pretrained_emotion_codec()
        model.enc["images"] = ImageBaseline.get_model(100)
        return model, state_dict



    def run(self, gpu=0):
        self.setup()

        logging.info(f"Image Proposed experiment run on {self.dataset_class} using pretrained ResNet.")

        logging.info("Building model...")

        model, orginal_state_dict = self.get_model()
        model.set_default(encoder="images", decoders=self.dataset_class.format)


        logging.info(f"{model}")

        logging.info("Setting up gpu...")
        gpu = torch.device(f"cuda:{gpu}")
        model.to(device=gpu)

        logging.info("Loading data...")
        train_transform = data.images.get_ResNet_Preprocessor(data_augmentation=True)
        test_transform = data.images.get_ResNet_Preprocessor(data_augmentation=False)
        ds = self.dataset_class(split="train", transform=train_transform)
        dl = DataLoader(dataset=ds, batch_size=self.train_batchsize, shuffle=True, num_workers=16)

        logging.info("Preparing training...")
        optimizer = torch.optim.SGD(model.enc["images"].parameters(), lr=self.initial_lr, momentum=0.9)
        logging.info(optimizer)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        logging.info(scheduler)
        loss_fns = {"pred": self.dataset_class.loss(reduction="mean"),
                    "embsim": torch.nn.MSELoss()}
        logging.info(loss_fns)
        model.dec.eval()

        logging.info("Starting to train...")
        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}. Current learning rate is {scheduler.get_lr()}.")

            model.enc["images"].train()
            model.enc[self.dataset_class.format].eval()
            model.dec.eval()

            epoch_loss = {"pred": torch.tensor(0., requires_grad=False),
                          "embsim": torch.tensor(0., requires_grad=False),
                          "total": torch.tensor(0., requires_grad=False)}
            for batch in dl:
                optimizer.zero_grad()

                features = batch["features"].to(gpu)
                labels = batch["labels"].to(gpu)

                emb_sample = model.enc(x=features, encoder="images")
                if labels.dtype == torch.int64:
                    # need to reformat labels to one-hot float, alternative would be to change mapping models to use embedding layers (mhmm, but then this would be differnt for regression or classed-based formats...)
                    input_labels = torch.nn.functional.one_hot(labels, num_classes=len(self.dataset_class.variables))
                    input_labels = input_labels.float()
                else:
                    input_labels = labels
                emb_label = model.enc(x=input_labels, encoder=self.dataset_class.format)

                preds = model.dec(x=emb_sample, decoders=self.dataset_class.format)

                pred_loss  = 1 * loss_fns["pred"](preds, labels)
                embsim_loss = 0 * loss_fns["embsim"](emb_sample, emb_label)
                total_loss = pred_loss + embsim_loss

                total_loss.backward()
                optimizer.step()
                epoch_loss["pred"] += pred_loss
                epoch_loss["embsim"] += embsim_loss
                epoch_loss["total"] += total_loss

            scheduler.step()
            # normalize epoch losses by number of batches (they are already normalized within batches by loss function)
            for k, v in epoch_loss.items():
                v = v / len(dl)
                v = v.item()
                epoch_loss[k] = v  # for some reason, .items() returns by value and not by reference...
            self.save_tensorboard_scalars("train_loss", epoch_loss, i_epoch)

            logging.info(f"Epoch_loss: {epoch_loss}")
            logging.info("Start validation")

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

            # data handling
            if isinstance(result, list):
                result = {var: result[i] for i, var in enumerate(self.dataset_class.variables + ["mean"])}
                self.save_tensorboard_scalars("devset_performance", result, i_epoch)
            else:
                self.save_tensorboard_scalars("devset_performance", result, i_epoch)

            logging.info(f"Perfomance in epoch {i_epoch}: {result}")
            self.save_results(i_epoch, result)
            self.save_tensorboard_scalars("loss", {"train": epoch_loss, "dev": dev_loss}, i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()
        self.end_experiment()




BASEPATH = utils.get_project_root() / "emocoder" / "target" / "image" / "proposed" / "dev"
TESTDIR = utils.get_project_root() / "emocoder" / "target" / "image" / "proposed" / "test"

EXPERIMENTS = {
    "fer_be": ImageProposed(name="fer_be",
                  parent_dir=BASEPATH,
                  dataset_class=data.images.FER2013,
                  epochs=42,
                  train_batchsize=32,
                  test_batchsize=128
                  ),
    "fer_vad": ImageProposed(name="fer_vad",
                  parent_dir=BASEPATH,
                  dataset_class=data.images.FER2013Vad,
                  epochs=42,
                  train_batchsize=32,
                  test_batchsize=128
                  ),
    # "iaps": ImageProposed(name="iaps",
    #               parent_dir=BASEPATH,
    #               dataset_class=data.images.IAPS2008,
    #               epochs=42,
    #               train_batchsize=32,
    #               test_batchsize=128,
    #               initial_lr=.01,
    #               scheduler_step_size=5,
    #               scheduler_gamma=.1),
    "affectnet_be": ImageProposed(name="affectnet_be",
                                  parent_dir=BASEPATH,
                                  dataset_class=data.images.AffectNet2019_BE,
                                  epochs=20,
                                  train_batchsize=32,
                                  test_batchsize=128),
    "affectnet_va": ImageProposed(name="affectnet_va",
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

        model, __ = dev_experiment.get_model()
        model.set_default(encoder="images", decoders=dev_experiment.dataset_class.format)

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
