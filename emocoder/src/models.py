import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
from itertools import chain
from typing import Union, List




class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError




class LinearDecoder(nn.Linear):
    def __init__(self, num_inputs:int, num_outputs:int):
        super().__init__(in_features=num_inputs, out_features=num_outputs, bias=False)

class UnipolarDecoder(Decoder):

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.linear = nn.Linear(in_features=num_inputs, out_features=num_outputs, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class BipolarDecoder(Decoder):

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.linear = nn.Linear(in_features=self.num_inputs, out_features=num_outputs, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return 2 * (self.sigmoid(self.linear(x)) - 0.5)  # "stretched sigmoid" to cover interval [-1; 1]






# ================ META DECODER ================ #

# class PoolingDecoder(nn.Module):
#     """
#     Meta decoder which runs multiple decoders in parallel and concats the input.
#     """

# Different classes have different meanings in different datasets, so it doesnt make a lot of sense to have for example
# one "joy"-class-decoder and use this across all datasets.

class NumericToDiscreteDecoder(nn.Module):

    def __init__(self,
                 primary_decoders, #list of m decoders with output size 1
                 mapping_matrix): # m*n torch tensor mapping n-dimensional output of the m decoders to n classes
        super().__init__()
        assert len(primary_decoders) == mapping_matrix.shape[0]
        self.mapping_matrix = torch.nn.Parameter(mapping_matrix, requires_grad=False)
        self.primary_decoders = torch.nn.ModuleList(primary_decoders)

    def forward(self, x):
        pred = torch.cat([dec(x) for dec in self.primary_decoders], dim=-1)
        pred = pred.unsqueeze(-1)
        classpreds = - ((pred - self.mapping_matrix)**2).mean(dim=1) # dims: 0:batch; 1:primary_dims, 2:classes
        return classpreds


class SubsetDecoder(nn.Module):
    """
    Takes existing decoder and returns only part of its output, e.g., VA of VAD.
    """
    def __init__(self, primary_decoder:torch.nn.Module, subset: List[int]):
        super().__init__()
        self.primary_decoder = primary_decoder
        self.subset = subset

    def forward(self, x):
        x = self.primary_decoder(x)
        x = x[:, self.subset]
        return x

# ================================================== #





class LatentLayer(nn.Module):
    def __init__(self, size, activation:torch.nn.Module = None, dropout:float=0.):
        """
        Dummy Module for latent representation
        :param size:
        :param activation:
        """
        super(LatentLayer, self).__init__()
        self.size = size
        self.activation = activation
        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout)
        elif dropout == 0:
            self.dropout = None
        else:
            raise ValueError

    def forward(self, x):
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class ModuleCollection(nn.ModuleDict):
    """
    Base class for EncoderCollection and DecoderCollection
    """

    def __init__(self, modules=None):
        super(ModuleCollection, self).__init__(modules)

    #  I would probably want to put the stuff that handels self.to/parameters/eval/train/save here...


class EncoderCollection(ModuleCollection):

    def __init__(self, modules=None, latent=None, latent_size=100):
        super(EncoderCollection, self).__init__(modules)

        if isinstance(latent, LatentLayer):
            self.inter = latent
        elif latent is None and isinstance(latent_size, int):
            self.inter = LatentLayer(latent_size)
        else:
            raise ValueError("Illegal specification for latent layer")



    def forward(self, x, encoder):
        x = self[encoder](x)
        x = self.inter(x)
        return x




class DecoderCollection(ModuleCollection):
    def __init__(self, modules=None):
        super(DecoderCollection, self).__init__(modules)

    def forward(self, x, decoders):
            return {dec: self[dec](x) for dec in decoders}
            #return torch.cat([self[dec](x) for dec in decoders], dim=1)


class EncoderBlock(nn.Module):

    def __init__(self, latent_size, latent_activation, latent_dropout):
        super().__init__()
        self.pool  = EncoderCollection()
        self.proj = LatentLayer(size=latent_size,
                                activation=latent_activation,
                                dropout=latent_dropout)

    def __getitem__(self, item):
        return self.pool[item]

    def __setitem__(self, key, value):
        self.pool[key] = value

    def __delitem__(self, key):
        del self.pool[key]

    def forward(self, x, encoder):
        encoder = self.pool[encoder]
        x = encoder(x)
        x = self.proj(x)
        return x



class DecoderBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = DecoderCollection()
        #self.formatters = ModuleCollection()

    def __getitem__(self, item):
        return self.pool[item]

    def __setitem__(self, key, value):
        self.pool[key] = value

    def __delitem__(self, key):
        del self.pool[key]

    def forward(self, x,
                decoders: Union[str, list]
                ):
        if isinstance(decoders, str):
            x = self.pool(x, decoders=[decoders])
            x = x[decoders]
            return x
        else:
            x = self.pool(x, decoders=decoders)
            return torch.cat([x[dec] for dec in decoders], dim=1)




class EmotionCodec(nn.Module):

    def __init__(self, latent_size, latent_activation, latent_dropout):
        '''

        :param encoder_pool: dict str -> torch.nn.module
        :param decoder_pool: dict str -> torch.nn.module.
        :param latent: torch.nn.module  # if something else it may be difficult to put on gpu
        '''

        super(EmotionCodec, self).__init__()
        self.enc = EncoderBlock(latent_size=latent_size,
                                latent_activation=latent_activation,
                                latent_dropout=latent_dropout)
        self.dec = DecoderBlock()
        self._default_encoder = None
        self._default_decoders = None

    @property
    def embedding_size(self):
        """
        :return: int. Number of units in the intermediate emotion representation.
        """
        return self.enc.proj.size


    def forward(self, x, encoder: str = None, decoders: Union[str, list] = None):
        if encoder is None: encoder = self.get_default_encoder()
        if decoders is None: decoders = self.get_default_decoders()
        x = self.enc(x, encoder)
        x = self.dec(x, decoders)
        return x

# TODO Data type of decoders should be changed to str only. Parameter should also be named "decoder", because we now have eg a VAD decoder and not three individual ones.
    def set_default(self, encoder: str = None, decoders: Union[str, list] = None):
        # checking input validity
        if encoder not in self.enc.pool.keys():
            raise ValueError(f"Encoder {encoder} not part of the model.")
        if decoders not in self.dec.pool.keys():
            raise ValueError(f"Decoder {decoders} not part of the model.")
        # setting defaults
        self._default_encoder = encoder
        self._default_decoders = decoders

    def select(self, encoder: str, decoders: Union[str, list]):
        self.set_default(encoder, decoders)
        return self

    def get_default_encoder(self):
        if self._default_encoder is None:
            raise ValueError
        else:
            return self._default_encoder

    def get_default_decoders(self):
        if self._default_decoders is None:
            raise ValueError
        else:
            return self._default_decoders


    def parameters(self, encoders=None, decoders=None, intermediate=False):

        if encoders is None and decoders is None and not intermediate:
            return super().parameters()
        else:
            iters = []
            if encoders:
                for x in encoders:
                    iters.append(self.enc[x].parameters())
            if intermediate:
                iters.append(self.enc.proj.parameters())
            if decoders:
                for x in decoders:
                    iters.append(self.dec[x].parameters())
            return chain.from_iterable(iters)

    def to(self, encoders: list=None, decoders: list=None, intermediate: bool=False, *args, **kwargs):
        if encoders is None and decoders is None and not intermediate:
            return super().to(*args, **kwargs)
        else:
            if encoders:
                for x in encoders:
                    self.enc[x] = self.enc[x].to(*args, **kwargs)
            if decoders:
                for x in decoders:
                    self.dec[x] = self.dec[x].to(*args, **kwargs)
            if intermediate:
                self.iter = self.iter.to(*args, **kwargs)


    def train(self, mode=True, encoders=None, decoders=None, intermediate=False):
        if encoders is None and decoders is None and not intermediate:
            super().train(mode)
        else:
            if encoders:
                for x in encoders:
                    self.enc[x].train(mode)
            if decoders:
                for x in decoders:
                    self.dec[x].train(mode)
            if intermediate:
                self.inter.train(mode)
            return self # this is according to the original torch code: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.train

    def eval(self, encoders=None, decoders=None, intermediate=False):
        return self.train(mode=False, encoders=encoders, decoders=decoders, intermediate=intermediate)



class MappingFFN(nn.Module):
    """
    COLING 2018 style FFN for mapping between emotion formats.
    """
    def __init__(self, num_inputs, num_outputs, scaling: Union[str, torch.nn.Module]):
        super(MappingFFN, self).__init__()

        assert scaling in ["tanh", "sigmoid", "logits"] or isinstance(scaling, torch.nn.Module)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(self.num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.num_outputs)

        self.activation = nn.LeakyReLU(.01)

        self.hidden_drop = nn.Dropout(p=.2)

        if scaling == "tanh":
            self.scaling = torch.nn.Tanh()
        elif scaling == "sigmoid":
            self.scaling = torch.nn.Sigmoid()
        elif scaling == "logits":
            self.scaling = None
        elif isinstance(scaling, torch.nn.Module):
            self.scaling = scaling
        else:
            raise ValueError

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.hidden_drop(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.hidden_drop(x)

        x = self.fc3(x)
        if self.scaling:
            x = self.scaling(x)

        return x



class WordFFN(nn.Module):

    def __init__(self, num_outputs, scaling: Union[str, torch.nn.Module], num_inputs=300, hidden_layers=(256, 128)):
        '''
        NAACL 2018 style FFN for word-level emotion prediction.
        :param num_outputs:
        '''
        super(WordFFN, self).__init__()


        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_layers)):

            # if first hidden layer make connection from inputs otherwise from last hidden layer
            if i == 0:
                self.hidden_layers.append(nn.Linear(num_inputs, hidden_layers[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            # if last hidden layer create connection to outputs
            if i== len(hidden_layers) - 1:
                self.hidden_layers.append(nn.Linear(hidden_layers[i], num_outputs))

        self.LReLU = nn.LeakyReLU(0.01)

        if scaling == "tanh":
            self.scaling = torch.nn.Tanh()
        elif scaling == "sigmoid":
            self.scaling = torch.nn.Sigmoid()
        elif scaling == "logits":
            self.scaling = None
        elif isinstance(scaling, torch.nn.Module):
            self.scaling = scaling
        else:
            raise ValueError

        self.drop_embs = nn.Dropout(p=0.2)
        self.drop_hidden = nn.Dropout(p=0.5)

    def forward(self, x, labels=None):
        # input layer
        x = self.drop_embs(x)

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if i < len(self.hidden_layers) - 1: # if not last transformation
                x = self.LReLU(x)
                x = self.drop_hidden(x)

        if self.scaling:
            x = self.scaling(x)

        return x


class GRU(nn.Module):
    def __init__(self, num_outputs,
                 fixed_length):
        super(GRU, self).__init__()
        self.fixed_length = fixed_length
        self.gru = nn.GRU(input_size=300, hidden_size=128, num_layers=1,
                          batch_first=True)


        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        self.drop_hidden = nn.Dropout(p=0.5)  # 0.5 dropout
        self.drop_vertical = nn.Dropout(p=0.5)  # 0.5 dropout at last timestep of gru-layer

    def forward(self, x):
        # print(x.shape)

        # GRU layer
        out, hidden = self.gru(x)
        x = hidden.squeeze()
        x = self.drop_vertical(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_hidden(x)

        # output layer
        x = self.fc2(x)

        return x

class BertSequencePooler(BertForSequenceClassification):
    """Basically the super class from Huggingface, but it only outputs logits and nothing else. Best used in Emotion Codec."""


    def forward(self, input_ids):

        logits = (super().forward(input_ids=input_ids))[0]

        return logits


def get_transformer_sequence_regressor(model_class=BertSequencePooler,
                                       tokenizer_class=BertTokenizer,
                                       pretrained_weights='bert-base-uncased',
                                       num_outputs=1):
    # bert-base-chinese

    model = model_class.from_pretrained(pretrained_weights, num_labels=num_outputs)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    return model, tokenizer



def get_resnet(variant=50, pretrained=True):
    assert variant in [18, 34, 50, 101, 152], 'variant of resnet must be 18, 34, 50, 101 or 152.'

    if variant == 18:
        return torchvision.models.resnet18(pretrained=pretrained)
    elif variant == 34:
        return torchvision.models.resnet34(pretrained=pretrained)
    elif variant == 50:
        return torchvision.models.resnet50(pretrained=pretrained)
    elif variant == 101:
        return torchvision.models.resnet101(pretrained=pretrained)
    elif variant == 152:
        return torchvision.models.resnet152(pretrained=pretrained)
