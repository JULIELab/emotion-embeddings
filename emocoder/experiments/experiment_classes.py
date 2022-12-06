import logging
from typing import Union

import torch


from emocoder.src import models, data
from emocoder.src.experiments import Experiment


class Checkpoint_Test_Mapping(Experiment):

    def __init__(self,
                 name: str,
                 parent_dir,
                 dataset_class: data.mapping.MappingDataset.__class__,
                 features_key,
                 labels_key,
                 split,
                 model,
                 checkpoint):
        super().__init__(name=name,
                         parent_dir=parent_dir,
                         performance_key=dataset_class.performance_key[labels_key],
                         greater_is_better=dataset_class.greater_is_better[labels_key])

        self.dataset_class =  dataset_class
        self.features_key = features_key
        self.labels_key = labels_key
        self.split = split
        self.model = model
        self.checkpoint = checkpoint

    def run(self, gpu):
        self.setup()
        logging.info("Folder structure set up")

        logging.info("Loading checkpoint, parameterizing model")
        state_dict = torch.load(self.checkpoint)
        self.model.load_state_dict(state_dict)
        gpu = torch.device(f"cuda:{gpu}")
        self.model.to(device=gpu)

        logging.info("Start testing (there is no training phase in this experiment)")
        metric, __ = self.dataset_class.score(features_key=self.features_key, labels_key=self.labels_key,
                                                    model=self.model, device=gpu, split=self.split)
        result = metric.result()
        logging.info(f"Result obtained in zero-shot scenario: {result}")
        self.save_results("zeroshot_perf", result)
        self.end_experiment()



class Checkpoint_Test_Text(Experiment):

    def __init__(self, name: str,
                 parent_dir,
                 dataset_class,
                 split,
                 model,
                 tokenizer,
                 checkpoint: str,
                 test_batchsize: int,
                 checkpoint_exact_match: bool = True):
        super().__init__(name=name, parent_dir=parent_dir, performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)
        self.dataset_class = dataset_class
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint
        self.checkpoint_exact_match = checkpoint_exact_match
        self.test_batchsize = test_batchsize
        self.split = split

    def run(self,gpu):
        self.setup()
        logging.info("Folder structure set up")

        logging.info("Building model, loading checkpoint...")
        state_dict = torch.load(self.checkpoint)
        self.model.load_state_dict(state_dict, strict=self.checkpoint_exact_match)
        gpu = torch.device(f"cuda:{gpu}")
        self.model.to(device=gpu)
        self.model.eval()

        logging.info("Preparing data...")

        transform, collater = data.utils.get_text_transform_and_collater(dataset_class=self.dataset_class,
                                                                         tokenizer=self.tokenizer)


        logging.info("Start testing (there is no training phase in this experiment)")
        metric, __ = self.dataset_class.score(self.model,
                                          device=gpu,
                                          split=self.split,
                                          batch_size=self.test_batchsize,
                                          collater=collater,
                                          transform=transform)

        result = metric.result()

        logging.info(f"Result obtained in zero-shot scenario: {result}")
        self.save_results("zeroshot_perf", result)
        self.end_experiment()


class Checkpoint_Test_Image(Experiment):

    def __init__(self, name: str,
                 parent_dir,
                 dataset_class,
                 split:str,
                 model,
                 checkpoint,
                 checkpoint_exact_match=True #whether load_state_dict should be run with "strict" flag
                 ):
        super().__init__(name=name, parent_dir=parent_dir, performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)
        self.dataset_class = dataset_class
        self.checkpoint = checkpoint
        self.checkpoint_exact_match = checkpoint_exact_match
        self.split = split
        self.model = model


    def run(self, gpu):
        self.setup()
        logging.info("Folder structure set up")

        logging.info("Loading checkpoint, parameterizing model")
        state_dict = torch.load(self.checkpoint)
        self.model.load_state_dict(state_dict, strict=self.checkpoint_exact_match)
        # self.model.set_default(encoder="images", decoders=self.dataset_class.format)
        gpu = torch.device(f"cuda:{gpu}")
        self.model.to(device=gpu)

        logging.info("Preparing data")
        test_transform = data.images.get_ResNet_Preprocessor(data_augmentation=False)

        logging.info("Start testing (there is no training phase in this experiment)")
        metric, loss = self.dataset_class.score(self.model, device=gpu, split=self.split, transform=test_transform)
        result = metric.result()

        logging.info(f"Result obtained in zero-shot scenario: {result}")
        self.save_results("zeroshot_perf", result)
        self.end_experiment()


class Checkpoint_Test_Word(Experiment):
    """
    Load pretrained word encoders for a particular dataset and decoders trained with mapping.
    Then test on other dataset applicable for given word encoder without further training. E.g., load XANEW
    word encoder and test on XAENW-BE and so forth.
    """

    def __init__(self, name: str,
                 parent_dir,
                 dataset_class,
                 split:str,
                 embeddings: Union[str, type(data.vectors.Embedding_Model)],
                 embedding_limit,
                 model,
                 checkpoint,
                 checkpoint_exact_match=True): #whether load_state_dict should be run with "strict" flag
        super().__init__(name=name, parent_dir=parent_dir, performance_key=dataset_class.performance_key,
                         greater_is_better=dataset_class.greater_is_better)
        self.dataset_class = dataset_class
        self.split = split
        self.embeddings = embeddings
        self.embedding_limit = embedding_limit
        self.model = model
        self.checkpoint = checkpoint
        self.checkpoint_exact_match = checkpoint_exact_match


    def run(self,gpu):
        self.setup()
        logging.info("Folder structure set up")

        logging.info("Loading checkpoint, parameterizing model")
        state_dict = torch.load(self.checkpoint)
        self.model.load_state_dict(state_dict, strict=self.checkpoint_exact_match)
        gpu = torch.device(f"cuda:{gpu}")
        self.model.to(device=gpu)

        logging.info("Preparing data")
        emb_transform = data.utils.Embedding_Lookup_Transform(embeddings=self.embeddings, limit=self.embedding_limit)

        logging.info("Start testing (there is no training phase in this experiment)")
        metric, test_loss = self.dataset_class.score(self.model, device=gpu, split=self.split, transform=emb_transform)
        result = metric.result()

        logging.info(f"Result obtained in zero-shot scenario: {result}")
        self.save_results("zeroshot_perf", result)
        self.end_experiment()