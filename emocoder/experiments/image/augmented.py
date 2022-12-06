import logging
from typing import Type
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from emocoder.src import utils, data, constants
from emocoder.src.experiments import Experiment, get_best_checkpoint
from  emocoder.experiments.mapping import multitask as multitask_mapping_experiment
from emocoder.experiments.image.baseline import ImageBaseline
from emocoder.experiments import utils as xutils, constants as xconstants, experiment_classes


class Augmented(Experiment):

    def __init__(self,
                 name:str,
                 parent_dir: Path,
                 dataset_class: Type[data.images.ImageDataset],
                 augmentation_format: str, #also the name of the decoder used to synthesize labels
                 augmentation_prediction_problem, # from src.constant; important for formatting the synthetic labels
                 epochs,
                 train_batchsize,
                 test_batchsize):
        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)

        # self.decoder = decoder
        self.dataset_class = dataset_class
        self.epochs = epochs
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize
        self.augmentation_format = augmentation_format
        self.augmentation_prediction_problem = augmentation_prediction_problem


    @staticmethod
    def get_model():
        model, state_dict = xutils.get_pretrained_emotion_codec()
        model.enc["images"] = ImageBaseline.get_model(100)
        return model, state_dict



    def run(self, gpu=0):
        self.setup()

        logging.info(f"Image Augmented experiment run on {self.dataset_class} using pretrained ResNet.")

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
        optimizer = torch.optim.SGD(model.enc["images"].parameters(), lr=0.001, momentum=0.9)
        logging.info(optimizer)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=.5)
        logging.info(scheduler)
        loss_fn = self.dataset_class.loss(reduction="mean")
        loss_fn_augmented = constants.FORMATS_2_LOSSES[self.augmentation_format](reduction="mean")
        logging.info(f"Regular loss function:{loss_fn}; Loss function for augmented labels: {loss_fn_augmented}.")

        logging.info("Starting to train...")
        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}. Current learning rate is {scheduler.get_lr()}.")

            model.enc["images"].train()
            model.enc[self.dataset_class.format].eval()
            model.dec.eval()

            # setting up training loss tracking
            epoch_loss = {"pred": torch.tensor(0.),
                          "aug": torch.tensor(0.),
                          "total": torch.tensor(0.)}

            for batch in dl:
                optimizer.zero_grad()

                features = batch["features"].to(gpu)
                labels = batch["labels"].to(gpu)

                # computing augmented labels
                # integer-based class-encodings must be converted to one-hot
                if self.dataset_class.problem == constants.MULTICLASS:
                    num_labels = len(self.dataset_class.variables)
                    input_labels = one_hot(labels, num_labels)
                    input_labels = input_labels.to(torch.float32)
                else:
                    input_labels = labels

                augmented_labels = model(x=input_labels, encoder=self.dataset_class.format, decoders=self.augmentation_format)
                # if the augmentation format is a multiclass problem, synthetic labels must be converted into one-hot format to mimic actual labels.
                # TODO What about data augmentation for multi-class-single-label formats? Shouldn't I have to re-format the synthetic labels too?
                # TODO Extract a general function for emotion label augmentation.

                if self.augmentation_prediction_problem == constants.MULTIVARIATE_REGRESSION:
                    pass
                elif self.augmentation_prediction_problem == constants.MULTILABEL: #reformat to one-hot
                    raise NotImplementedError()
                elif self.augmentation_prediction_problem == constants.MULTICLASS: # reformat to integer-based
                    augmented_labels = torch.argmax(augmented_labels, dim=1)
                else:
                    raise ValueError("Must specify a prediction probelem from src.constants.py")




                embs = model.enc(features, encoder="images")
                preds = model.dec(embs, decoders=self.dataset_class.format)
                augmented_preds = model.dec(embs, decoders=self.augmentation_format)
                prediction_loss = loss_fn(preds, labels)
                augmentation_loss = loss_fn_augmented(augmented_preds, augmented_labels)
                loss =  prediction_loss + augmentation_loss

                epoch_loss["pred"] += prediction_loss
                epoch_loss["aug"] += augmentation_loss
                epoch_loss["total"] += loss


                loss.backward()
                optimizer.step()

            scheduler.step()

            # normalize epoch losses by number of batches (they are already normalized within batches by loss function)
            for k, v in epoch_loss.items():
                v = v / len(ds)
                v = v.item()
                epoch_loss[k] = v

            logging.info(f"Train loss in epoch {i_epoch}: {epoch_loss}")
            self.save_tensorboard_scalars("train_loss", epoch_loss, i_epoch)

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
            self.save_tensorboard_scalars("dev_loss", dev_loss, i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()
        self.end_experiment()




BASEPATH = utils.get_project_root() / "emocoder" / "target" / "image" / "augmented" / "dev"
TESTDIR = utils.get_project_root() / "emocoder" / "target" / "image" / "augmented" / "test"

EXPERIMENTS = {
    "fer_be": Augmented(name="fer_be",
                  parent_dir=BASEPATH,
                  dataset_class=data.images.FER2013,
                  augmentation_format="vad",
                  augmentation_prediction_problem=constants.MULTIVARIATE_REGRESSION,
                  epochs=42,
                  train_batchsize=32,
                  test_batchsize=128),
    "fer_vad": Augmented(name="fer_vad",
                  parent_dir=BASEPATH,
                  dataset_class=data.images.FER2013Vad,
                  augmentation_format="be_fer13",
                  augmentation_prediction_problem=constants.MULTICLASS,
                  epochs=42,
                  train_batchsize=32,
                  test_batchsize=128),
    # "iaps": Augmented(name="iaps",
    #               parent_dir=BASEPATH,
    #               dataset_class=data.images.IAPS2008,
    #               augmentation_format="be5",
    #               augmentation_prediction_problem=constants.MULTIVARIATE_REGRESSION,
    #               epochs=42,
    #               train_batchsize=32,
    #               test_batchsize=128),
    "affectnet_be": Augmented(name="affectnet_be",
                      parent_dir=BASEPATH,
                      dataset_class=data.images.AffectNet2019_BE,
                      augmentation_format="va",
                      augmentation_prediction_problem=constants.MULTIVARIATE_REGRESSION,
                      epochs=20,
                      train_batchsize=32,
                      test_batchsize=128),
    "affectnet_va": Augmented(name="affectnet_va",
                              parent_dir=BASEPATH,
                              dataset_class=data.images.AffectNet2019_VA,
                              augmentation_format="be_affectnet",
                              augmentation_prediction_problem=constants.MULTICLASS,
                              epochs=20,
                              train_batchsize=32,
                              test_batchsize=128),

}
# TODO extend interface of augmented experiments in other modulalities with `augmentation_prediction_problem` as well.


def run_all_dev_exp(gpu=0):
    for x in EXPERIMENTS.values():
        x.run(gpu=gpu)

def run_all_test_exp(gpu):
    for exp_name, dev_experiment in EXPERIMENTS.items():
        checkpoint = get_best_checkpoint(utils.get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name))

        model, __ = dev_experiment.get_model()
        model.set_default(encoder="images", decoders=dev_experiment.dataset_class.format)

        test_experiment = experiment_classes.Checkpoint_Test_Image(name=dev_experiment.base_name,
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