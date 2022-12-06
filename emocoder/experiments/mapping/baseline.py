import logging

import argparse
import torch
from torch.utils.data import DataLoader

from emocoder.src import models, data, metrics, constants, utils
from emocoder.src.experiments import Experiment,get_best_checkpoint
from emocoder.src.utils import get_experiment_dir
from emocoder.experiments import constants as xconstants, experiment_classes

class MappingBaseline(Experiment):

    def __init__(self,
                 name,
                 parent_dir,
                 dataset_class: data.mapping.MappingDataset.__class__,
                 features_key,
                 labels_key,
                 batch_size,
                 epochs):
        """

        :param name: Base name of the experiment. Will appear in path name to the produced experimental data.
        :param parent_dir: Parent directory of
        :param dataset_class:
        :param features_key:
        :param labels_key:
        :param batch_size:
        :param epochs:
        """
        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key[labels_key],
                         greater_is_better=dataset_class.greater_is_better[labels_key])

        self.dataset_class = dataset_class
        self.features_key = features_key
        self.labels_key = labels_key

        self.batch_size = batch_size
        self.epochs = epochs

    @staticmethod
    def get_model(num_inputs, num_outputs, scaling):
        return models.MappingFFN(num_inputs=num_inputs,
                                 num_outputs=num_outputs,
                                 scaling=scaling) # everything is logits now...

    def run(self, gpu):

        self.setup()

        logging.info(f"Mapping experiment run on {self.dataset_class} using {self.features_key} as features and "
                     f"{self.labels_key} as labels_key")

        logging.info("Building model")
        model = self.get_model(num_inputs=len(self.dataset_class.variables[self.features_key]),
                               num_outputs=len(self.dataset_class.variables[self.labels_key]),
                               scaling="logits")
        gpu = torch.device(f"cuda:{gpu}")
        model.to(gpu)

        logging.info("Loading training data")
        ds = self.dataset_class(split="train")
        dl = DataLoader(dataset=ds, batch_size=self.batch_size)

        logging.info("preparing training")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        loss_fn = self.dataset_class.loss[self.labels_key]()

        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}")
            model.train()
            epoch_loss = torch.tensor(0.)
            for batch in dl:
                optimizer.zero_grad()
                features = batch[f"features_{self.features_key}"].to(gpu)
                labels = batch[f"labels_{self.labels_key}"].to(gpu)
                preds = model(features)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            epoch_loss = epoch_loss.item()
            logging.info(f"Loss in epoch {i_epoch}: {epoch_loss}")

            logging.info("Start validation")
            metric, dev_loss = self.dataset_class.score(features_key=self.features_key, labels_key=self.labels_key,
                                              model=model, device=gpu, split="dev")

            if not self.performance_order_defined:
               self.order_performance_according_to(metric)

            result = metric.result()
            logging.info(f"Performance in epoch {i_epoch}: {result}")

            self.save_results(i_epoch, result)
            self.save_tensorboard_scalars("devset_performance", result, i_epoch)
            self.save_tensorboard_scalars("train_loss", epoch_loss, i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()


TARGETDIR = xconstants.MAPPING_BASELINE_BASEPATH
DEVDIR = TARGETDIR / "dev"
TESTDIR = TARGETDIR / "test"


experiments = {
    "anew_stevenson_baseline_vad_to_be5": MappingBaseline(name="anew_stevenson_baseline_vad_to_be5",
                             parent_dir=DEVDIR,
                             dataset_class=data.mapping.ANEW_Stevenson,
                             features_key="vad",
                             labels_key="be5",
                             batch_size = 32,
                             epochs=50),
    "anew_stevenson_baseline_be5_to_vad": MappingBaseline(name="anew_stevenson_baseline_be5_to_vad",
                                                          parent_dir=DEVDIR,
                                                          dataset_class=data.mapping.ANEW_Stevenson,
                                                          features_key="be5",
                                                          labels_key="vad",
                                                          batch_size=32,
                                                          epochs=50),
    "anew_stevenson_baseline_va_to_be5": MappingBaseline(name="anew_stevenson_baseline_va_to_be5",
                                                          parent_dir=DEVDIR,
                                                          dataset_class=data.mapping.ANEW_VA_Stevenson,
                                                          features_key="va",
                                                          labels_key="be5",
                                                          batch_size=32,
                                                          epochs=50),
    "anew_stevenson_baseline_be5_to_va": MappingBaseline(name="anew_stevenson_baseline_be5_to_va",
                                                          parent_dir=DEVDIR,
                                                          dataset_class=data.mapping.ANEW_VA_Stevenson,
                                                          features_key="be5",
                                                          labels_key="va",
                                                          batch_size=32,
                                                          epochs=50),
    "xanew_nrc_baseline_vad_to_nrc": MappingBaseline(name="xanew_nrc_baseline_vad_to_nrc",
                             parent_dir=DEVDIR,
                             dataset_class=data.mapping.XANEW_NRC,
                             features_key="vad",
                             labels_key="nrc",
                             batch_size=32,
                             epochs=50),
    "xanew_nrc_baseline_nrc_to_vad": MappingBaseline("xanew_nrc_baseline_nrc_to_vad",
                             parent_dir=DEVDIR,
                             dataset_class=data.mapping.XANEW_NRC,
                             features_key="nrc",
                             labels_key="vad",
                             batch_size=32,
                             epochs=50),
    "FER_BE_VAD": MappingBaseline("FER-BE-VAD",
                         parent_dir=DEVDIR,
                         dataset_class=data.mapping.FER_BE_VAD,
                         features_key="be_fer13",
                         labels_key="vad",
                         batch_size=32,
                         epochs=50),
    "FER_VAD_BE": MappingBaseline("FER-VAD-BE",
                                  parent_dir=DEVDIR,
                                  dataset_class=data.mapping.FER_BE_VAD,
                                  features_key="vad",
                                  labels_key="be_fer13",
                                  batch_size=32,
                                  epochs=50),
    "AffectNet_BE_VA": MappingBaseline("AffectNet-BE-VA",
                                  parent_dir=DEVDIR,
                                  dataset_class=data.mapping.AffectNet_Mapping,
                                  features_key="be_affectnet",
                                  labels_key="va",
                                  batch_size=32,
                                  epochs=50),
    "AffectNet_VA_BE": MappingBaseline("AffectNet-VA-BE",
                                       parent_dir=DEVDIR,
                                       dataset_class=data.mapping.AffectNet_Mapping,
                                       features_key="va",
                                       labels_key="be_affectnet",
                                       batch_size=32,
                                       epochs=50),
}


def run_all_dev_exp(gpu):
    for x in experiments.values():
        x.run(gpu)

def run_all_test_exp(gpu):
    for dev_experiment in experiments.values():

        checkpoint = get_best_checkpoint(get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name+"-"))
        model = dev_experiment.get_model(num_inputs=len(dev_experiment.dataset_class.variables[dev_experiment.features_key]),
                               num_outputs=len(dev_experiment.dataset_class.variables[dev_experiment.labels_key]),
                               scaling="logits")

        test_experiment = experiment_classes.Checkpoint_Test_Mapping(name=dev_experiment.base_name,
                                                                     parent_dir=TESTDIR,
                                                                     dataset_class=dev_experiment.dataset_class,
                                                                     features_key=dev_experiment.features_key,
                                                                     labels_key=dev_experiment.labels_key,
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
