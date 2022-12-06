import argparse

from emocoder.src import data, models, experiments, constants, metrics, utils
from torch.utils.data import DataLoader
import logging
import torch
from random import choice
from emocoder.experiments import constants as xconstants, experiment_classes
from emocoder.experiments.mapping import baseline
from emocoder.src.experiments import get_best_checkpoint
from emocoder.src.utils import get_experiment_dir, nested_dict_update

from typing import List
from pathlib import Path
import numpy as np

class Exp(experiments.Experiment):

    def __init__(self,
                 name: str,
                 parent_dir: Path,
                 dataset_classes: List[data.mapping.MappingDataset.__class__],
                 # features_key: str,
                 # labels_key: str,
                 batch_size: int,
                 epochs: int,
                 loss_weights: dict = None
                 ):
        """
        Note that this experiment class does not have a feautures_key and labels_key parameter (in contrast to
        mapping/baseline or mapping/proposed. This is because in this experiment both mapping directions are addressed
        as part of the same experiment (hence the name 'multitask').

        :param name:
        :param parent_dir:
        :param dataset_classes:
        :param batch_size:
        :param epochs:
        :param loss_weights: A hierarchical dictionary (type - dataset - format) that *updates* the default loss.
        See `self.loss_weights` for illustration.
        """
        super().__init__(name=name, parent_dir=parent_dir, performance_key="overall_mean", greater_is_better=True)

        self.dataset_classes = dataset_classes
        self.formats: List[str] = [] # keys from constants
        for dsc in self.dataset_classes:
            for frmt in dsc.format:
                    self.formats.append(frmt)
        self.formats = list(set(self.formats))
        self.batch_size = batch_size
        self.epochs = epochs

        # build default loss weights (1.0 everywhere) and update with argument
        self.loss_weights = {"embsim": 1.0,
                             "mapping": {dsc.__name__: {frmt: 1.0 for frmt in dsc.format} for dsc in self.dataset_classes},
                             "autoencode": {dsc.__name__: {frmt: 1.0 for frmt in dsc.format} for dsc in self.dataset_classes},
                             "decodercosine": 0.1,
                             }
        if loss_weights: self.loss_weights = nested_dict_update(self.loss_weights, loss_weights)


    def get_model(self):
        """
        In the multitask variant, the models gets an encoder and a decoder for every emotion format used
        in a dataset class in self.dataset_classes.
        """
        model = models.EmotionCodec(latent_size=100, latent_activation=None, latent_dropout=.0)

        for format in self.formats:
            num_vars = len(constants.VARIABLES[format])
            model.enc[format] = models.MappingFFN(num_inputs=num_vars,
                                                             num_outputs=100,
                                                             scaling=constants.LOGITS)

            ### Version where all decoders have linear activation
            model.dec[format] = models.LinearDecoder(num_inputs=100,
                                                     num_outputs=num_vars)

            ### Version where some decoders have sigmoid avtivation
            # if format in ["va","vad"]:
            #     model.dec[format] = models.BipolarDecoder(num_inputs=100, num_outputs=num_vars)
            # elif format in ["be5"]:
            #     model.dec[format] = models.UnipolarDecoder(num_inputs=100, num_outputs=num_vars)
            # elif format in ["be_fer13", "be_affectnet"]:
            #     model.dec[format] = models.LinearDecoder(num_inputs=100, num_outputs=num_vars)
            # else:
            #     raise ValueError(f"Format {format} not handled!")

        # If both VAD and Va decoders exist, replace VA decoder by subsampled VAD encoder.
        ### replacing this with soft parameter sharing
        # if "va" in self.formats and "vad" in self.formats:
        #     model.dec["va"] = models.SubsetDecoder(primary_decoder=model.dec["vad"], subset=[0,1])
        #     assert model.dec["va"].primary_decoder is model.dec["vad"]

        return model

    def _get_decoder_cosine_loss_input(self, model):
        """
        Utility method that returns the input needed to compute torch.nn.CosineEmbeddingLoss: x1 and x2, tensors holding
        all head parameters (variables in  axis 0), y tensor of 1/-1 indicating whether these parameters should agree
        with each other.
        :return:
        """

        # 4-tuples of format1, varname1, format2, varname2
        heads_to_align = [("vad", "valence", "va", "valence"), # va - vad # both the same parameter now, so this give 0 loss
                          ("vad", "arousal", "va", "arousal"),
                          #("vad", "valence", "be5", "joy"), # vad - be5, maybe not a perfect match, but lets roll with it for now
                          #("va", "valence", "be5", "joy"), # va - be5
                          #("vad", "valence", "be_fer13", "happy"), # vad - be_fer13
                          #("va", "valence", "be_fer13", "happy"), # va - be_fer13
                          #("vad", "valence", "be_affectnet", "happiness"), # vad - be_affectnet
                          #("va", "valence", "be_affectnet", "happiness"), # va - be_affectnet
                          ("be5", "joy", "be_fer13", "happy"), # be5 - be_fer13
                          ("be5", "anger", "be_fer13", "anger"),
                          ("be5", "sadness", "be_fer13", "sad"),
                          ("be5", "fear", "be_fer13", "fear"),
                          ("be5", "disgust", "be_fer13", "disgust"),
                          ("be5", "joy", "be_affectnet", "happiness"),
                          ("be5", "anger", "be_affectnet", "anger"),
                          ("be5", "sadness", "be_affectnet", "sadness"),
                          ("be5", "fear", "be_affectnet", "fear"),
                          ("be5", "disgust", "be_affectnet", "disgust"),
                          ("be_fer13", "anger", "be_affectnet", "anger"), # be_fer13 - be_affectnet
                          ("be_fer13", "disgust", "be_affectnet", "disgust"),
                          ("be_fer13", "fear", "be_affectnet", "fear"),
                          ("be_fer13", "happy", "be_affectnet", "happiness"),
                          ("be_fer13", "sad", "be_affectnet", "sadness"),
                          ("be_fer13", "surprise", "be_affectnet", "surprise"),
                          ("be_fer13", "neutral", "be_affectnet", "neutral")]
        x1 = []
        x2 = []
        for i in range(len(heads_to_align)):
            # if len(heads_to_align[i]) != 4:
            #     raise ValueError(f"Malformatted in data in row {i}")
            format1, var1, format2, var2 = heads_to_align[i]

            # getting indices of vars in respective formats
            j1 = constants.VARIABLES[format1].index(var1)
            j2 = constants.VARIABLES[format2].index(var2)

            x1.append(model.dec[format1].weight[j1])
            x2.append(model.dec[format2].weight[j2])

        x1 = torch.stack(x1)
        x2 = torch.stack(x2)

        y = torch.ones(len(heads_to_align))

        return x1, x2,y


    def run(self, gpu):
        self.setup()
        logging.info(f"Setting up experiment to train VAD and BE5 decoders with representaiton mapping...")

        logging.info("Building model...")
        model = self.get_model()
        logging.info(model)

        logging.info("Setting up GPU")
        gpu = torch.device(f'cuda:{gpu}')
        model.to(device=gpu)

        logging.info("Preparing data")
        dataloaders = []


        for dsc in self.dataset_classes:
            ds = dsc("train")
            dl = DataLoader(dataset=ds,
                            batch_size=32,
                            shuffle=True,
                            num_workers=0)
            dataloaders.append(dl)




        logging.info("preparing training")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

        losses = {}
        for dsc in self.dataset_classes:
            for frmt in dsc.format:
                losses[(dsc, frmt)] = dsc.loss[frmt]()
        losses["embsim"] = torch.nn.MSELoss()
        losses["decodercosine"] = torch.nn.CosineEmbeddingLoss()

        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}")
            model.train()

            # setting up training loss tracking (which is pretty complicated in this case)
            epoch_loss = {}
            for dsc in self.dataset_classes:
                for frmt in dsc.format:
                    epoch_loss[f"mapping_{dsc.__name__}_{frmt}"] = torch.tensor(0., requires_grad=False)
                    epoch_loss[f"autoencode_{dsc.__name__}_{frmt}"] = torch.tensor(0., requires_grad=False)
            epoch_loss["embsim"] = torch.tensor(0., requires_grad=False)
            epoch_loss["decodercosine"] = torch.tensor(0., requires_grad=False)
            epoch_loss["total"] = torch.tensor(0., requires_grad=False)


            dl = data.utils.MultiDataLoaderIterator2(dataloaders, 100)
            for batch in dl:
                optimizer.zero_grad()
                dsc = batch["dataloader"].dataset.__class__

                format_1 = dsc.format[0]
                format_2 = dsc.format[1]
                    
                features_1 = batch[f"features_{format_1}"].to(gpu)
                features_2 = batch[f"features_{format_2}"].to(gpu)
                labels_1 = batch[f"labels_{format_1}"].to(gpu)
                labels_2 = batch[f"labels_{format_2}"].to(gpu)

                emb_1 = model.enc(x=features_1, encoder=format_1)
                emb_2 = model.enc(x=features_2, encoder=format_2)

                preds_1_1 = model.dec(x=emb_1, decoders=format_1)
                preds_1_2 = model.dec(x=emb_1, decoders=format_2)
                preds_2_1 = model.dec(x=emb_2, decoders=format_1)
                preds_2_2 = model.dec(x=emb_2, decoders=format_2)

                #retrieving loss weights
                w_embsim = self.loss_weights["embsim"]
                w_map_f1 = self.loss_weights["mapping"][dsc.__name__][format_1]
                w_map_f2 = self.loss_weights["mapping"][dsc.__name__][format_2]
                w_auto_f1 = self.loss_weights["autoencode"][dsc.__name__][format_1]
                w_auto_f2 = self.loss_weights["autoencode"][dsc.__name__][format_2]

                # Computing losses with weights
                mapping_loss_1_2 = w_map_f2 * losses[(dsc, format_2)](preds_1_2, labels_2)
                mapping_loss_2_1 = w_map_f1 * losses[(dsc, format_1)](preds_2_1, labels_1)
                autoencode_loss_1 = w_auto_f1 * losses[(dsc, format_1)](preds_1_1, labels_1)
                autoencode_loss_2 = w_auto_f2 * losses[(dsc, format_2)](preds_2_2, labels_2)
                embsim_loss = w_embsim * losses["embsim"](emb_1, emb_2)
                # TODO This copying back and forth is probably a bit slow. I may want to optimize this for speed later.
                decodercosine_loss = losses["decodercosine"](*[x.to(gpu) for x in self._get_decoder_cosine_loss_input(model)])
                loss = autoencode_loss_1 + autoencode_loss_2 + mapping_loss_1_2 + mapping_loss_2_1 + embsim_loss + decodercosine_loss

                epoch_loss[f"mapping_{dsc.__name__}_{format_1}"] += mapping_loss_2_1
                epoch_loss[f"mapping_{dsc.__name__}_{format_2}"] += mapping_loss_1_2
                epoch_loss[f"autoencode_{dsc.__name__}_{format_1}"] += autoencode_loss_1
                epoch_loss[f"autoencode_{dsc.__name__}_{format_2}"] += autoencode_loss_2
                epoch_loss["embsim"] += embsim_loss
                epoch_loss["decodercosine"] += decodercosine_loss
                epoch_loss["total"] += loss

                loss.backward()
                optimizer.step()

            # normalize epoch losses by number of batches (they are already normalized within batches by loss function)
            for k, v in epoch_loss.items():
                v = v / len(dl)
                v = v.item()
                epoch_loss[k] = v # for some reason, .items() returns by value and not by reference...
            self.save_tensorboard_scalars("train_loss", epoch_loss, i_epoch)

            logging.info(f"Epoch_loss: {epoch_loss}")
            logging.info("Start validation")

            # For every dataset test both ways
            metrics = {}
            dev_losses = {}
            for dsc in self.dataset_classes:

                format_1 = dsc.format[0]
                format_2 = dsc.format[1]

                key_1_2 = f"{dsc.__name__}_{format_1}_{format_2}"
                key_2_1 = f"{dsc.__name__}_{format_2}_{format_1}"

                metrics[key_1_2], dev_losses[key_1_2] = dsc.score(features_key=format_1,
                                                                                   labels_key=format_2,
                                                                                   device=gpu,
                                                                                   model=model.select(
                                                                                       encoder=format_1,
                                                                                       decoders=format_2),
                                                                                   split="dev")
                metrics[key_2_1], dev_losses[key_2_1] = dsc.score(features_key=format_2,
                                                                                   labels_key=format_1,
                                                                                   device=gpu,
                                                                                   model=model.select(
                                                                                       encoder=format_2,
                                                                                       decoders=format_1),
                                                                                   split="dev")
            results = {key: metric.result() for key,metric in metrics.items()}

            for ds_key, ds_result in results.items():
                # prefix every key in ds_result with dataset key
                ds_result = {f"{ds_key}_{var_key}": val for var_key, val in ds_result.items()}
                self.save_tensorboard_scalars("devset_performance",ds_result , i_epoch)
            results[self.performance_key] = np.mean([metric.result()[metric.performance_key] for metric in metrics.values()])
            logging.info(results)
            self.save_tensorboard_scalars("devset_performance", {self.performance_key: results["overall_mean"]}, i_epoch)

            self.save_results(i_epoch, results)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()
        self.end_experiment()



TARGET_PATH = xconstants.MAPPING_MULTITASK_BASEPATH / "dev"
TEST_PATH =  xconstants.MAPPING_MULTITASK_BASEPATH / "test"


EXPERIMENTS = {
    # "full": Exp(name="full",
    #                          parent_dir=TARGET_PATH,
    #                          dataset_classes=[data.mapping.ANEW_Stevenson,
    #                                           data.mapping.ANEW_VA_Stevenson,
    #                                           data.mapping.XANEW_NRC,
    #                                           data.mapping.FER_BE_VAD,
    #                                           data.mapping.AffectNet_Mapping
    #                                           ],
    #                          batch_size = 32,
    #                          epochs = 50),
    "no_nrc": Exp(name="no_nrc",
                parent_dir=TARGET_PATH,
                dataset_classes=[data.mapping.ANEW_Stevenson,
                                 data.mapping.ANEW_VA_Stevenson,
                                 #data.mapping.XANEW_NRC,
                                 data.mapping.FER_BE_VAD,
                                 data.mapping.AffectNet_Mapping
                                 ],
                batch_size=32,
                epochs=100,
                loss_weights= {"embsim": 1.,
                               "decodercosine": 1.,
                               "mapping": {data.mapping.FER_BE_VAD.__name__: {"be_fer13":1./10.,
                                                                              #"vad": 1./2.
                                                                              },
                                           data.mapping.AffectNet_Mapping.__name__: {"be_affectnet": 1./10.,
                                                                                     #"va": 1./2.
                                                                                     }
                                           },
                               "autoencode": {data.mapping.FER_BE_VAD.__name__: {"be_fer13": 1./10.},
                                              data.mapping.AffectNet_Mapping.__name__: {"be_affectnet": 1./10.}},
                               },
                  ),
    # "no_fer": Exp(name="no_fer",
    #               parent_dir=TARGET_PATH,
    #               dataset_classes=[data.mapping.ANEW_Stevenson,
    #                                data.mapping.ANEW_VA_Stevenson,
    #                                data.mapping.XANEW_NRC,
    #                                #data.mapping.FER_BE_VAD,
    #                                data.mapping.AffectNet_Mapping
    #                                ],
    #               batch_size=32,
    #               epochs=50),
    # "no_affectnet": Exp(name="no_affectnet",
    #               parent_dir=TARGET_PATH,
    #               dataset_classes=[data.mapping.ANEW_Stevenson,
    #                                data.mapping.ANEW_VA_Stevenson,
    #                                data.mapping.XANEW_NRC,
    #                                data.mapping.FER_BE_VAD,
    #                                #data.mapping.AffectNet_Mapping
    #                               ],
    #               batch_size=32,
    #               epochs=50),
    # "no_fer_affectnet": Exp(name="no_fer_affectnet",
    #                     parent_dir=TARGET_PATH,
    #                     dataset_classes=[data.mapping.ANEW_Stevenson,
    #                                      data.mapping.ANEW_VA_Stevenson,
    #                                      data.mapping.XANEW_NRC,
    #                                      # data.mapping.FER_BE_VAD,
    #                                      # data.mapping.AffectNet_Mapping
    #                                      ],
    #                     batch_size=32,
    #                     epochs=50),
    # "no_nrc_fer_affectnet": Exp(name="no_nrc_fer_affectnet",
    #                         parent_dir=TARGET_PATH,
    #                         dataset_classes=[data.mapping.ANEW_Stevenson,
    #                                          data.mapping.ANEW_VA_Stevenson,
    #                                          # data.mapping.XANEW_NRC,
    #                                          # data.mapping.FER_BE_VAD,
    #                                          # data.mapping.AffectNet_Mapping
    #                                          ],
    #                         batch_size=32,
    #                         epochs=50),


}


def run_all_dev_exp(gpu):
    for name, exp in EXPERIMENTS.items():
        exp.run(gpu)

def run_all_test_exp(gpu):

    dev_experiment = EXPERIMENTS["no_nrc"]
    model = dev_experiment.get_model()
    checkpoint = get_best_checkpoint(get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name))

    for dataset_class in dev_experiment.dataset_classes:

        model.set_default(encoder=dataset_class.format[0], decoders=dataset_class.format[1])
        test_experiment = experiment_classes.Checkpoint_Test_Mapping(name=f"{dataset_class.__name__}_{dataset_class.format[0]}_{dataset_class.format[1]}" ,
                                                                     parent_dir=TEST_PATH,
                                                                     dataset_class=dataset_class,
                                                                     features_key=dataset_class.format[0],
                                                                     labels_key=dataset_class.format[1],
                                                                     split="test",
                                                                     model=model,
                                                                     checkpoint=checkpoint)
        test_experiment.run(gpu)

        # Other way round
        model.set_default(encoder=dataset_class.format[1], decoders=dataset_class.format[0])
        test_experiment = experiment_classes.Checkpoint_Test_Mapping(name=f"{dataset_class.__name__}_{dataset_class.format[1]}_{dataset_class.format[0]}",
                                                                     parent_dir=TEST_PATH,
                                                                     dataset_class=dataset_class,
                                                                     features_key=dataset_class.format[1],
                                                                     labels_key=dataset_class.format[0],
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

