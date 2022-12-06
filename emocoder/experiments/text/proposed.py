import argparse
from torch.utils.data import DataLoader
import logging
import torch
from transformers.optimization import  AdamW

from emocoder.src import data, models, experiments, utils
from emocoder.experiments import utils as xutils, constants as xconstants, experiment_classes
from emocoder.src.experiments import get_best_checkpoint
from emocoder.src.utils import get_experiment_dir


class Exp(experiments.Experiment):
    """
    Load pretrained BE5 and VAD decoders. Then, fit a encoder for a specific dataset.
    """

    def __init__(self,
                 name:str,
                 parent_dir,
                 dataset_class,
                 pretrained_weights: str,# hugging faces model names; 'bert-base-uncased' for English, bert-base-chinese for Chinese
                 epochs,
                 train_batchsize,
                 test_batchsize):
        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)
        self.dataset_class = dataset_class
        self.pretrained_weights = pretrained_weights
        self.epochs =epochs
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize




    @staticmethod
    def get_model(pretrained_weights):
        """
        :param pretrained_weights: hugging faces model names; e.g.,  'bert-base-uncased' for English, bert-base-chinese for Chinese
        :return: tuple model, tokenizer, state_dict
        """
        model, state_dict = xutils.get_pretrained_emotion_codec()
        model.enc["text"], tokenizer = models.get_transformer_sequence_regressor(
            pretrained_weights=pretrained_weights,
            num_outputs=model.embedding_size)
        return model, tokenizer, state_dict




    def run(self,gpu):
        self.setup()
        logging.info("Folder structure set up")
        logging.info("Building model, loading parameters...")
        model, tokenizer, original_state_dict = self.get_model(self.pretrained_weights)


        gpu = torch.device(f"cuda:{gpu}")
        model.to(device=gpu)
        model.set_default(encoder="text", decoders=self.dataset_class.format)

        logging.info("Preparing data...")
        transform, collater = data.utils.get_text_transform_and_collater(dataset_class=self.dataset_class,
                                                                         tokenizer=tokenizer)
        ds = self.dataset_class("train", transform=transform)
        dl = DataLoader(dataset=ds,
                        batch_size=self.train_batchsize,
                        shuffle=True,
                        num_workers=16,
                        collate_fn=collater)

        logging.info("Preparing optimization...")
        optimizer = AdamW(params=model.enc["text"].parameters(), lr=1e-5)
        loss_fn = self.dataset_class.loss()

        logging.info("Starting training loop...")
        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}")
            model.enc["text"].train()
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
            logging.info(f"Loss in epoch {i_epoch}: {epoch_loss}")
            epoch_loss = epoch_loss.item()

            # dev set eval
            metric, val_loss = self.dataset_class.score(model, device=gpu, split="dev", batch_size=self.test_batchsize,
                                              collater=collater,
                                              transform=transform)
            result = metric.result()
            logging.info(f"Perfomance in epoch {i_epoch}: {result}")

            self.save_results(i_epoch, result)
            self.save_tensorboard_scalars("devset_performance", result, i_epoch)

            self.save_tensorboard_scalars("loss", {"train": epoch_loss, "dev": val_loss}, i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()

        self.compare_model_with_state_dict(model, original_state_dict)
        self.end_experiment()


PARENTDIR = xconstants.TEXT_PROPOSED_BASEPATH / "dev"
TESTDIR = xconstants.TEXT_PROPOSED_BASEPATH / "test"

experiment_list = [

    # Exp(name="sst2cls",
    #     parent_dir=PARENTDIR,
    #     dataset_class=data.text.SST_2_Class,
    #     pretrained_weights="bert-base-uncased",
    #     epochs=20,
    #     train_batchsize=24,
    #     test_batchsize=48),

    Exp(name="emobank",
        parent_dir=PARENTDIR,
        dataset_class=data.text.EmoBank,
        pretrained_weights="bert-base-uncased",
        epochs=20,
        train_batchsize=24,
        test_batchsize=48),

    Exp(name="cvat",
        parent_dir=PARENTDIR,
        dataset_class=data.text.CVAT,
        pretrained_weights="bert-base-chinese",
        epochs=20,
        train_batchsize=12,
        test_batchsize=24),

    Exp(name="afftext",
        parent_dir=PARENTDIR,
        dataset_class=data.text.AffTextBE5,
        pretrained_weights="bert-base-uncased",
        epochs=40,
        train_batchsize=24,
        test_batchsize=48),

]



def run_all_dev_exp(gpu):
    for x in experiment_list:
        x.run(gpu)

def run_all_test_exp(gpu):
    for dev_experiment in experiment_list:


        checkpoint = get_best_checkpoint(get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name))

        model, tokenizer, __ = dev_experiment.get_model(pretrained_weights=dev_experiment.pretrained_weights)
        model.set_default(encoder="text", decoders=dev_experiment.dataset_class.format)

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

