from emocoder.src import data, models, experiments, constants, metrics, utils
from torch.utils.data import DataLoader
import logging
import torch
from random import choice
from emocoder.experiments import constants as xconstants
from emocoder.experiments.mapping import baseline

class Exp(experiments.Experiment):

    def __init__(self,
                 name,
                 parent_dir,
                 dataset_class: data.mapping.MappingDataset.__class__,
                 features_key,
                 labels_key,
                 batch_size,
                 epochs):
        super().__init__(name=name, parent_dir=parent_dir)

        self.dataset_class = dataset_class
        self.features_key = features_key
        self.labels_key = labels_key

        self.batch_size = batch_size
        self.epochs = epochs

        self.performance_key = self.dataset_class.performance_key[labels_key]
        self.greater_is_better = self.dataset_class.greater_is_better[labels_key]



    def get_model(self):
        model = models.EmotionCodec(latent_size=100, latent_activation=None, latent_dropout=.0)
        model.enc[self.features_key] = models.MappingFFN(num_inputs=len(self.dataset_class.variables[self.features_key]),
                                                         num_outputs=100,
                                                         scaling="logits")

        model.dec[self.labels_key] = models.LinearDecoder(num_inputs=100,
                                                       num_outputs=len(self.dataset_class.variables[self.labels_key]))

        model.set_default(encoder=self.features_key, decoders=self.labels_key)

        return model



    def run(self):
        self.setup()
        logging.info(f"Setting up experiment to train VAD and BE5 decoders with representaiton mapping...")

        logging.info("Building model...")
        model = self.get_model()


        ds = self.dataset_class("train")
        dl = DataLoader(dataset=ds,
                              batch_size=32,
                              shuffle=True,
                              num_workers=4)


        logging.info("Setting up GPU")
        gpu = torch.device('cuda:0')
        model.to(device=gpu)

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
                preds = model(x=features, encoder=self.features_key, decoders=self.labels_key)
                loss = loss_fn(preds, labels)

                loss.backward()
                optimizer.step()
                epoch_loss += loss

            epoch_loss = epoch_loss.item() / len(dl)
            logging.info("Start validation")


            metric, dev_loss = self.dataset_class.score(features_key=self.features_key,
                                              labels_key=self.labels_key,
                                              model=model,
                                              device=gpu,
                                              split="dev")

            result = metric.result()


            logging.info(f"Performance in epoch {i_epoch}: {result}")
            self.save_results(i_epoch, result)
            self.save_tensorboard_scalars("loss", {"train_loss": epoch_loss, "dev_loss": dev_loss}, i_epoch)
            self.save_tensorboard_scalars("devset_performance", result, i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()


TARGET_PATH = xconstants.MAPPING_PROPOSED_BASEPATH


EXPERIMENTS = {
    "anew_stevenson_baseline_vad_to_be5": Exp(name="anew_stevenson",
                             parent_dir=TARGET_PATH,
                             dataset_class=data.ANEW_Stevenson,
                             features_key="vad",
                             labels_key="be5",
                             batch_size = 32,
                             epochs = 50),
    "anew_stevenson_baseline_be5_to_vad": Exp(name="stevenson-anew",
                                                          parent_dir=TARGET_PATH,
                                                          dataset_class=data.ANEW_Stevenson,
                                                          features_key="be5",
                                                          labels_key="vad",
                                                          batch_size=32,
                                                          epochs=50),
    "anew_stevenson_baseline_va_to_be5": Exp(name="anewVA-stevenson",
                                                          parent_dir=TARGET_PATH,
                                                          dataset_class=data.ANEW_VA_Stevenson,
                                                          features_key="va",
                                                          labels_key="be5",
                                                          batch_size=32,
                                                          epochs=50),
    "anew_stevenson_baseline_be5_to_va": Exp(name="stevenson-anewVA",
                                                          parent_dir=TARGET_PATH,
                                                          dataset_class=data.ANEW_VA_Stevenson,
                                                          features_key="be5",
                                                          labels_key="va",
                                                          batch_size=32,
                                                          epochs=50),
    "xanew_nrc_baseline_vad_to_nrc": Exp(name="xanew-nrc",
                             parent_dir=TARGET_PATH,
                             dataset_class=data.XANEW_NRC,
                             features_key="vad",
                             labels_key="nrc",
                             batch_size=32,
                             epochs=50),
    "xanew_nrc_baseline_nrc_to_vad": Exp("nrc-xanew",
                             parent_dir=TARGET_PATH,
                             dataset_class=data.XANEW_NRC,
                             features_key="nrc",
                             labels_key="vad",
                             batch_size=32,
                             epochs=50),
    "FER_BE_VAD": Exp("FerBE-FerVAD",
                         parent_dir=TARGET_PATH,
                         dataset_class=data.mapping.FER_BE_VAD,
                         features_key="be_fer13",
                         labels_key="vad",
                         batch_size=32,
                         epochs=50),
    "FER_VAD_BE": Exp("FerVAD-FerBE",
                                  parent_dir=TARGET_PATH,
                                  dataset_class=data.mapping.FER_BE_VAD,
                                  features_key="vad",
                                  labels_key="be_fer13",
                                  batch_size=32,
                                  epochs=50),
    "AffectNet_BE_VA": Exp("AffectnetBE-AffectnetVA",
                                  parent_dir=TARGET_PATH,
                                  dataset_class=data.mapping.AffectNet_Mapping,
                                  features_key="be_affectnet",
                                  labels_key="va",
                                  batch_size=32,
                                  epochs=50),
    "AffectNet_VA_BE": Exp("AffectnetVA-AffectnetBE",
                                       parent_dir=TARGET_PATH,
                                       dataset_class=data.mapping.AffectNet_Mapping,
                                       features_key="va",
                                       labels_key="be_affectnet",
                                       batch_size=32,
                                       epochs=50),
}


def run_all_dev_exp():
    global name, exp
    for name, exp in EXPERIMENTS.items():
        exp.run()


if __name__ == "__main__":
    run_all_dev_exp()


