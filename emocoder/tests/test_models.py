from emocoder.src import data, models
import torch
from torch.utils.data import DataLoader, random_split
from transformers.optimization import  AdamW
from emocoder.src.models import WordFFN
import scipy.stats as st
import numpy as np
from random import choice
import time
import torch

def test_get_transformer_sequence_regressor():

    model, tokenizer = models.get_transformer_sequence_regressor(num_outputs=3)

    # test very simple forward pass
    input_text = "A great first sentence!"
    encoded = tokenizer.encode(input_text)
    input_ids = torch.tensor([encoded])
    tmp = model(input_ids)
    assert isinstance(tmp, torch.Tensor)

    # get EmoBank dataset and dataloader
    eb_train = data.text.EmoBank('train', transform=data.utils.Tokenize_Transformer(tokenizer))
    eb_test = data.text.EmoBank('test', transform=data.utils.Tokenize_Transformer(tokenizer))
    tmp = eb_train[0]
    assert tmp
    coll = data.utils.Collater(padding_symbol=tokenizer.pad_token_type_id, num_labels=3, label_dtype=torch.float32)
    eb_train_loader = DataLoader(dataset=eb_train, batch_size=12, shuffle=True, num_workers=1,collate_fn=coll)
    # eb_test_loader = DataLoader(dataset=eb_test, batch_size=12, shuffle=True, num_workers=1, collate_fn=coll)
    for i_batch, batch in enumerate(eb_train_loader):
        assert batch
        model(batch['features'])
        break

def test_gpu_training():
    model, tokenizer = models.get_transformer_sequence_regressor(num_outputs=3)
    eb_train = data.text.EmoBank('train', transform=data.utils.Tokenize_Transformer(tokenizer))
    coll = data.utils.Collater(padding_symbol=tokenizer.pad_token_id, num_labels=3, label_dtype=torch.float32)
    train_loader = DataLoader(dataset=eb_train, batch_size=12, shuffle=True, num_workers=4, collate_fn=coll)

    loss_fn = torch.nn.MSELoss()

    EPOCHS = 1

    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters())


    for i in range(EPOCHS):

        epoch_loss = 0

        for i_batch, batch in enumerate(train_loader):
            optimizer.zero_grad()
            labels = batch['labels']
            features = batch['features']
            features, labels = features.to(device), labels.to(device) #watch out, the .to-method works differently for tensors and modules
            preds = model(features)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            break

def test_WordFFN():
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings="FB_CC_EN", limit=10*1000)
    xanew = data.words.XANEW('train', transform=emb_transform)
    loader = DataLoader(dataset=xanew, batch_size=128, shuffle=True, num_workers=12)

    model = models.WordFFN(3, scaling=data.words.XANEW.scaling)
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    model.to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    epochs = 1

    for i_epoch in range(epochs):

        #training loop
        model.train()
        for i_batch, batch in enumerate(loader):
            optimizer.zero_grad()
            features, labels = batch['features'], batch['labels']
            features, labels = features.to(device), labels.to(device)
            preds = model(features)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            break


def test_MappingFFN():
    train = data.mapping.ANEW_Stevenson("train")
    test = data.mapping.ANEW_Stevenson("dev")

    train_dataloader = DataLoader(dataset=train, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=test, batch_size=218, shuffle=False, num_workers=1)

    model = models.MappingFFN(3,5, scaling=data.mapping.ANEW_Stevenson.scaling["be5"])
    device = torch.device('cuda:0')
    model.to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    EPOCHS = 50

    def train():
        model.train()
        for i_batch, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            vad, be5 = batch['labels_vad'].to(device), batch['labels_be5'].to(device)
            preds = model(vad)
            loss = loss_fn(preds, be5)
            loss.backward()
            optimizer.step()

    def eval():
        model.eval()
        with torch.no_grad():
            last_perf = None
            for batch in test_dataloader:
                vad, be5 = batch['labels_vad'].to(device), batch['labels_be5'].to(device)
                preds = model(vad).cpu().numpy()
                actual = be5.cpu().numpy()
                last_perf = np.mean([st.pearsonr(preds[:,i], actual[:,i])[0] for i in range(actual.shape[1])])
        return last_perf


    print(eval())
    last_perf = None
    for i_epoch in range(EPOCHS):
        train()
        last_perf = eval()
        print(last_perf)
    assert last_perf > .7

def test_Emotion_Codec_interface():

    def iter_length(it):
        length = 0
        for i in it:
            length += 1
        return length

    model = models.EmotionCodec(latent_size=10,  latent_activation=None, latent_dropout=.0)
    model.enc["enc1"] = models.MappingFFN(3,10, scaling="logits")
    model.enc["enc2"] = models.MappingFFN(3,10, scaling="logits")
    model.dec["dec1"] = models.LinearDecoder(10, 1)
    model.dec["dec2"] = models.LinearDecoder(10, 1)

    gpu = torch.device('cuda:0')
    cpu = torch.device('cpu')

    # getting parameters
    len1 = iter_length(model.parameters(encoders=["enc1"]))
    len2 = iter_length(model.enc["enc1"].parameters())
    assert len1 == len2

    len1 = iter_length(model.parameters())
    len2 = iter_length(model.parameters(encoders=['enc1', 'enc2'],
                                        decoders=['dec1', 'dec2'],
                                        intermediate=True))
    assert len1 == len2

    params = model.parameters(encoders=['enc2'],
                              decoders=['dec1'],
                              intermediate=True)
    for p in params:
        assert isinstance(p, torch.nn.Parameter)

    del params
    # copying stuff to gpu and back

    ## dummy test to see if this kind of assertion makes sense
    model = model.to(device=gpu)
    for p in model.parameters():
        assert p.device == gpu
    model = model.to(device=cpu)
    for p in model.parameters():
        assert p.device == cpu
        # assert p.device == gpu # assert to false

    ## actual test
    model.to(device=gpu, encoders=["enc1"], decoders=["dec2"])
    for p in model.parameters(encoders=["enc2"], decoders=["dec1"]):
        assert p.device == cpu
    for p in model.parameters(encoders=["enc1"], decoders=["dec2"]):
        assert p.device == gpu

    del p
    model = model.to(device=cpu)

    model.train()
    assert model.training
    model.eval()
    assert not model.training

    model.train(encoders=["enc1", "enc2"])
    assert model.enc["enc1"].training
    assert model.enc["enc2"].training
    assert not model.dec["dec1"].training
    assert not model.dec["dec2"].training



def test_EmotionCodec_training():
    torch.manual_seed(0)

    # getting the model: encoders for english words, vad and be5, decoders for vad and be5

    # model = models.EmotionCodec(encoder_pool={}, decoder_pool={})
    model = models.EmotionCodec(latent_size=100, latent_activation=None, latent_dropout=.0)

    model.enc["vad"] =  models.MappingFFN(3, 100, scaling="logits")
    model.enc["be5"] =  models.MappingFFN(5, 100, scaling="logits")
    model.enc["word"] =  models.WordFFN(100, scaling="logits")

    model.dec["vad"] = models.LinearDecoder(100, 3)
    model.dec["be5"]= models.LinearDecoder(100, 5)

    # getting the data: anew-stevenson mapping dataset and respective word datasets

    # train, test = random_split(data.ANEW_Stevenson(), lengths=[800, 228])
    train = data.mapping.ANEW_Stevenson("train")
    test = data.mapping.ANEW_Stevenson("dev")
    train_dataloader_mapping = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader_mapping = DataLoader(dataset=test, batch_size=218, shuffle=False, num_workers=1)



    # Putting things to the gpu

    gpu = torch.device('cuda:0')
    cpu = torch.device('cpu')
    model.to(device=gpu)

    loss_fn = torch.nn.MSELoss()
    params = model.parameters()
    optimizer = torch.optim.Adam(params=params,
                                 lr=1e-3)



    # TRAINING ON MAPPING DATASET
    for i_epoch in range(20):

        model.train()
        for i_batch, batch in enumerate(train_dataloader_mapping):

            optimizer.zero_grad()

            # randomly choose an input format
            input_format = choice(data.mapping.ANEW_Stevenson.format)
            for key in batch.keys():
                batch[key] = batch[key].to(gpu)


            emb = model.enc(x=batch[f"features_{input_format}"], encoder=input_format)
            pred_vad = model.dec(emb, decoders="vad")
            pred_be5 = model.dec(emb, decoders="be5")

            # computing loss with augmented labels
            loss = loss_fn(batch['labels_vad'], pred_vad) + loss_fn(batch["labels_be5"], pred_be5)
            loss.backward()
            optimizer.step()

        ## eval
        model.eval()
        with torch.no_grad():
            last_perf = None
            for batch in test_dataloader_mapping:
                actual_vad, actual_be5 = batch['labels_vad'].to(gpu), batch['labels_be5'].to(gpu)

                preds_be5 = model(x=actual_vad, encoder="vad", decoders="be5").cpu().numpy()
                preds_vad = model(x=actual_be5, encoder="be5", decoders="vad").cpu().numpy()

                actual_be5 = actual_be5.cpu().numpy()
                actual_vad = actual_vad.cpu().numpy()

                last_perf_vad = np.mean([st.pearsonr(preds_vad[:, i], actual_vad[:, i])[0] for i in range(actual_vad.shape[1])])
                last_perf_be5 = np.mean([st.pearsonr(preds_be5[:, i], actual_be5[:, i])[0] for i in range(actual_be5.shape[1])])

            print(last_perf_vad, last_perf_be5)

    # Finished mapping pretraining, now on to fitting a word encoder...
    emb_transform = data.utils.Embedding_Lookup_Transform(embeddings="FB_CC_EN", limit=20 * 1000)
    train = data.words.ANEW1999("train", transform=emb_transform)
    test = data.words.ANEW1999("dev", transform=emb_transform)
    train_dataloader_anew = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader_anew = DataLoader(dataset=test, batch_size=218, shuffle=False, num_workers=1)

    test = data.words.Stevenson2007("dev", transform=emb_transform)
    test_dataloader_steve = DataLoader(dataset=test, batch_size=218, shuffle=False, num_workers=1)

    # keep decoders constant and only fit word encoder
    optimizer = torch.optim.Adam(params= model.enc["word"].parameters(), lr=1e-3)


    print("\nStart training word encoder \n")
    perf = []
    for i_epoch in range(20):
        model.train()
        for batch in train_dataloader_anew:
            optimizer.zero_grad()
            features, labels = batch['features'].to(gpu), batch['labels'].to(gpu)

            # mapping for data augmentation
            be5_mapped = model(x=labels, encoder="vad", decoders="be5")
            emb = model.enc(x=features, encoder="word")
            preds_vad = model.dec(emb, decoders="vad")
            preds_be5 = model.dec(emb, decoders="be5")


            loss = loss_fn(labels, preds_vad) + loss_fn(be5_mapped, preds_be5)
            loss.backward()
            optimizer.step()



        # test loop including be5 evaluation
        model.eval()
        with torch.no_grad():
            perf = []
            for batch in test_dataloader_anew:
                features, labels = batch['features'].to(gpu), batch['labels'].to(gpu)
                preds = model(x=features, encoder="word", decoders="vad")
                preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
                perf.extend([st.pearsonr(preds[:, i], labels[:, i])[0] for i in range(labels.shape[1])] )

            for batch in test_dataloader_steve:
                features, labels = batch['features'].to(gpu), batch['labels'].to(gpu)
                preds = model(x=features, encoder="word", decoders="be5")
                preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
                try:
                    perf.extend( [st.pearsonr(preds[:, i], labels[:, i])[0] for i in range(labels.shape[1])] )
                except ValueError:
                    perf.append(np.nan)
            print(perf)
    for x in perf:
        assert x > .2


def test_bert_regressor():
    model, tokenizer = models.get_transformer_sequence_regressor(num_outputs=3)
    eb_train = data.text.EmoBank('train', transform=data.utils.Tokenize_Transformer(tokenizer))
    coll = data.utils.Collater(padding_symbol=tokenizer.pad_token_id, num_labels=3, label_dtype=torch.float32)
    train_loader = DataLoader(dataset=eb_train, batch_size=24, shuffle=True, num_workers=16, collate_fn=coll)
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.MSELoss()
    losses = []

    for i_epoch in range(1):
        model.train()
        for i_batch, batch in enumerate(train_loader):
            optimizer.zero_grad()
            labels = batch['labels']
            features = batch['features']
            features, labels = features.to(device), labels.to(
                device)  # watch out, the .to-method works differently for tensors and modules
            preds = model(features)
            loss = loss_fn(preds, labels)
            losses.append(loss.cpu())

            loss.backward()
            optimizer.step()

        losses = torch.stack(losses)
        half = int(len(losses)/2)
        first_half, second_half = losses[:half].mean(), losses[half:].mean()

        assert first_half > second_half



