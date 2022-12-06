import argparse
import numpy as np
from emocoder.src import data, experiments, utils
from torch.utils.data import DataLoader
import logging
import torch
from typing import Union
from emocoder.experiments.word.proposed import Word_Proposed
from emocoder.experiments import constants as xconstants, experiment_classes
from emocoder.src.experiments import get_best_checkpoint
from emocoder.src.utils import get_experiment_dir


class Fit_Word_Encoders_on_Multiple_Datasets(experiments.Experiment):
    """Load pretrained decoders for VAD and BE5. Fit word encoders for a particular language on
    multiple datatsets in parallel. Test on both in parallel.

    This is meant as a comparison to `fit_word-encoders_after_mapping` and `test-word-encoders-zero-shot`.
    """

    def __init__(self,
                 name:str,
                 parent_dir,
                 dataset_classes,
                 epochs,
                 embeddings: Union[str, type(data.vectors.Embedding_Model)],
                 embedding_limit):

        super().__init__(name=name,
                         parent_dir=parent_dir,
                         greater_is_better=True,
                         performance_key="overall")
        self.epochs =epochs
        self.dataset_classes = dataset_classes
        self.embeddings = embeddings
        self.embedding_limit = embedding_limit


    @staticmethod
    def get_model():
        return Word_Proposed.get_model()

    def run(self,gpu=0):
        self.setup()
        logging.info("Folder structure set up")

        logging.info("Building model, loading decoder checkpoint")
        model, original_state_dict = self.get_model()
        gpu = torch.device(f"cuda:{gpu}")
        model.to(device=gpu)

        logging.info("Preparing data")
        emb_transform = data.utils.Embedding_Lookup_Transform(embeddings=self.embeddings, limit=self.embedding_limit)
        datasets = []
        dataloaders = []
        for dsc in self.dataset_classes:
            ds = dsc("train", transform=emb_transform, scale=True)
            dl = DataLoader(dataset=ds, shuffle=True, batch_size=128)
            datasets.append(ds)
            dataloaders.append(dl)

        logging.info("Preparing training")
        optimizer = torch.optim.Adam(params=model.enc.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        model.dec.eval()

        for i_epoch in range(self.epochs):
            logging.info(f"Beginning training in epoch {i_epoch}.")
            model.enc.train()
            epoch_loss = torch.tensor(0.)
            for batch in data.utils.MultiDataLoaderIterator(dataloaders):
                optimizer.zero_grad()
                features = batch["features"].to(gpu)
                labels = batch["labels"].to(gpu)
                targets = batch["dataloader"].dataset.format
                preds = model(x=features, encoder="words", decoders=targets)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            logging.info(f"Loss in epoch {i_epoch}: {epoch_loss.item()}")
            results = {}
            overall = []
            for ds in self.dataset_classes:
                model.set_default(encoder="words", decoders=ds.format)
                metric, val_loss = ds.score(model, device=gpu, split="dev", transform=emb_transform)
                result = metric.result()
                results[ds.__name__] = result
                overall.append(result[ds.performance_key])
            results[self.performance_key] = np.mean(overall)

            logging.info(f"Perfomance in epoch {i_epoch}: {results}")

            self.save_results(i_epoch, results)
            self.save_tensorboard_scalars("devset_performance", results, i_epoch)


            self.save_tensorboard_scalars("loss", epoch_loss.item(), i_epoch)
            self.save_checkpoint(model, f"model_{i_epoch}")
            self.remove_all_but_best_checkpoint()

        self.compare_model_with_state_dict(model, original_state_dict)
        self.end_experiment()

PARENTDIR = xconstants.WORD_MULTITASK_BASEPATH / "dev"
TESTDIR = xconstants.WORD_MULTITASK_BASEPATH / "test"

experiment_list = [
    # Fit_Word_Encoders_on_Multiple_Datasets(name="Fit-Word-Encoder-on-English",
    #                                        parent_dir=PARENTDIR,
    #                                        dataset_classes=[data.words.XANEW, data.words.XANEW_BE],
    #                                        embeddings=data.vectors.Facebook_CommonCrawl_English,
    #                                        embedding_limit=None,
    #                                        epochs=70),
    Fit_Word_Encoders_on_Multiple_Datasets(name="Fit-Word-Encoder-on-English",
                                           parent_dir=PARENTDIR,
                                           dataset_classes=[data.words.ANEW1999, data.words.Stevenson2007],
                                           embeddings=data.vectors.Facebook_CommonCrawl_English,
                                           embedding_limit=None,
                                           epochs=70),
    Fit_Word_Encoders_on_Multiple_Datasets(name="Fit-Word-Encoder-on-Spanish",
                                           parent_dir=PARENTDIR,
                                           dataset_classes=[data.words.Stadthagen_VA, data.words.Stadthagen_BE],
                                           embeddings=data.vectors.Facebook_CommonCrawl_Spanish,
                                           embedding_limit=None,
                                           epochs=70),
    Fit_Word_Encoders_on_Multiple_Datasets(name="Fit-Word-Encoder-on-German",
                                           parent_dir=PARENTDIR,
                                           dataset_classes=[data.words.Vo, data.words.Briesemeister],
                                           embeddings=data.vectors.Facebook_CommonCrawl_German,
                                           embedding_limit=None,
                                           epochs=70),
    Fit_Word_Encoders_on_Multiple_Datasets(name="Fit-Word-Encoder-on-Polish",
                                           parent_dir=PARENTDIR,
                                           dataset_classes=[data.words.Riegel, data.words.Wierzba],
                                           embeddings=data.vectors.Facebook_CommonCrawl_Polish,
                                           embedding_limit=None,
                                           epochs=70),
    Fit_Word_Encoders_on_Multiple_Datasets(name="Fit-Word-Encoder-on-Turkish",
                                           parent_dir=PARENTDIR,
                                           dataset_classes=[data.words.Kapucu_VA, data.words.Kapucu_BE],
                                           embeddings=data.vectors.Facebook_CommonCrawl_Turkish,
                                           embedding_limit=None,
                                           epochs=70),
]



def run_all_dev_exp(gpu=0):
    for x in experiment_list:
        x.run(gpu)

def run_all_test_exp(gpu):
    for dev_experiment in experiment_list:

        model, __ = dev_experiment.get_model()
        checkpoint = get_best_checkpoint(get_experiment_dir(dev_experiment.parent_dir, dev_experiment.base_name))

        for dataset_class in dev_experiment.dataset_classes:
            model.set_default(encoder="words", decoders=dataset_class.format)
            test_experiment = experiment_classes.Checkpoint_Test_Word(name=dataset_class.__name__,
                                                                      parent_dir=TESTDIR,
                                                                      dataset_class=dataset_class,
                                                                      split="test",
                                                                      embeddings=dev_experiment.embeddings,
                                                                      embedding_limit=dev_experiment.embedding_limit,
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