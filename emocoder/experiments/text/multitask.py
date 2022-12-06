import argparse
import numpy as np
from torch.utils.data import DataLoader
import logging
import torch
from transformers.optimization import  AdamW

from emocoder.src import data, models, experiments, utils
from emocoder.experiments.text import proposed
from emocoder.experiments import constants as xconstants, experiment_classes
from emocoder.src.experiments import get_best_checkpoint
from emocoder.src.utils import get_experiment_dir


class Exp(experiments.Experiment):
    """
    Load pretrained BE5 and VAD decoders. Then, fit a encoder for a specific dataset.
    """

    def __init__(self,
                 name:str,
                 parent_dir,
                 dataset_classes,
                 loss_weights, # factors to multiply dataset specific losses by
                 pretrained_weights: str,# hugging faces model names; 'bert-base-uncased' for English, 'bert-base-chinese' for Chinese
                 epochs,
                 train_batchsize,
                 test_batchsize):

        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key="overall",
                         greater_is_better=True)

        self.dataset_classes = dataset_classes
        self.loss_weights = loss_weights
        self.pretrained_weights = pretrained_weights
        self.epochs =epochs
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize


    @staticmethod
    def get_model(pretrained_weights):
        model, tokenizer, state_dict = proposed.Exp.get_model(pretrained_weights)
        return model, tokenizer, state_dict



    def run(self, gpu):
        self.setup()
        logging.info("Folder structure set up")

        logging.info("Building model, loading parameters...")
        model, tokenizer, original_state_dict = self.get_model(self.pretrained_weights)

        gpu = torch.device(f"cuda:{gpu}")
        model.to(device=gpu)

        logging.info("Preparing data...")
        transform = data.utils.Tokenize_Transformer(tokenizer=tokenizer)

        #### START: TRAINING ON MULTIPLE DATASETS ####
        datasets = []
        dataloaders = []
        collaters = {} #Maps from dataset class to collater
        transforms = {} # data transforms are also dataset dependent *sigh*
        loss_weights = {self.dataset_classes[i]: self.loss_weights[i] for i in range(len(self.dataset_classes))}

        for dsc in self.dataset_classes:

            transform, coll = data.utils.get_text_transform_and_collater(dataset_class=dsc,
                                                                             tokenizer=tokenizer)
            ds = dsc("train", transform=transform)
            dl = DataLoader(dataset=ds,
                        batch_size=self.train_batchsize,
                        shuffle=True,
                        num_workers=16,
                        collate_fn=coll)

            datasets.append(ds)
            collaters[dsc] = coll
            transforms[dsc] = transform
            dataloaders.append(dl)


        logging.info("Preparing optimization...")
        optimizer = AdamW(params=model.enc["text"].parameters(), lr=1e-5)
        loss_fns = {dsc: dsc.loss() for dsc in self.dataset_classes} #maps from dataset class to loss operator

        logging.info("Starting training loop...")
        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}")
            model.enc["text"].train()
            epoch_loss = torch.tensor(0.)
            for batch in data.utils.MultiDataLoaderIterator(dataloaders):
                optimizer.zero_grad()
                features = batch["features"].to(gpu)
                labels = batch["labels"].to(gpu)
                targets = batch["dataloader"].dataset.format
                preds = model(x=features, encoder="text", decoders=targets)
                current_dataset_class = batch["dataloader"].dataset.__class__
                current_loss_fn = loss_fns[current_dataset_class]
                current_loss_weight = loss_weights[current_dataset_class]
                loss = current_loss_weight * current_loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            epoch_loss = epoch_loss.item()
            logging.info(f"Loss in epoch {i_epoch}: {epoch_loss}")

            # dev set eval
            results = {}
            losses = {}
            for dsc in self.dataset_classes:
                model.set_default(encoder="text", decoders=dsc.format)
                metric, loss = dsc.score(model, device=gpu, split="dev", batch_size=self.test_batchsize,
                                              collater=collaters[dsc],
                                              transform=transform)
                result = metric.result()
                results[dsc.__name__] = result
                losses[dsc.__name__] = loss
                # TODO Write a class for multitask experiments independet of their modality! (I just ran into the problem, that outcomes are formatted differently in the image-multitask-experiment)

            #compute over all result
            results[self.performance_key] = np.mean([results[dsc.__name__][dsc.performance_key] for dsc in self.dataset_classes])

            logging.info(f"Perfomance in epoch {i_epoch}: {results}")
            self.save_results(i_epoch, results)
            self.save_tensorboard_scalars("devset_performance", results, i_epoch)

            self.save_tensorboard_scalars("loss", {"train": epoch_loss, "dev": losses}, i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()


        self.compare_model_with_state_dict(model, original_state_dict)
        self.end_experiment()


PARENTDIR = xconstants.TEXT_MULTITASK_BASEPATH / "dev"
TESTDIR = xconstants.TEXT_MULTITASK_BASEPATH / "test"

experiment_list = [

    # Very difficult becuase of domain mismatch and different loss function (different magnitude of loss values)
    Exp(name="emobank-afftext",
        parent_dir=PARENTDIR,
        dataset_classes=[data.text.EmoBank, data.text.AffTextBE5],
        loss_weights = [1., 1.],
        pretrained_weights="bert-base-uncased",
        epochs=20,
        train_batchsize=24,
        test_batchsize=48),

    # Exp(name="sst-emobank",
    #     parent_dir=PARENTDIR,
    #     dataset_classes=[data.text.SST_2_Class, data.text.EmoBank],
    #     loss_weights = [0.05, 1.],
    #     pretrained_weights="bert-base-uncased",
    #     epochs=20,
    #     train_batchsize=24,
    #     test_batchsize=48),
]


def run_all_dev_exp(gpu):
    for x in experiment_list:
        x.run(gpu)


def run_all_test_exp(gpu):
    for dev_experiment in experiment_list:


        checkpoint = get_best_checkpoint(get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name))

        model, tokenizer, __ = dev_experiment.get_model(pretrained_weights=dev_experiment.pretrained_weights)

        for dataset_class in dev_experiment.dataset_classes:
            model.set_default(encoder="text", decoders=dataset_class.format)
            test_experiment = experiment_classes.Checkpoint_Test_Text(name=dataset_class.__name__,
                                                                      parent_dir=TESTDIR,
                                                                      dataset_class=dataset_class,
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




