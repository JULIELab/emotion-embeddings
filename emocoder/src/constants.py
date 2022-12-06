import torch


# sets of emotion variables
VA = ["valence", "arousal"]
VAD = VA +["dominance"]
BE5 = ["joy", "anger", "sadness", "fear", "disgust"]
BE6 = BE5 + ["surprise"]
PLUTCHIK = BE6 + ["anticipation", "trust"]
IZARD = BE5 + ["shame", "guilt"]
POL = ["positive", "negative"]
POL1 = ["polarity"]
NRC = PLUTCHIK + POL
POL5 = ["very negative", "slightly negative", "neutral", "slightly positive", "very positive"]
BE_FLICKR = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
BE_FER13 = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
BE_AFFECTNET = ["neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger", "contempt"]

# These are the original affectnet classes, but we remove the no-emotion categories.
# BE_AffectNet = ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'no-face']



# Ugly workaround because I need global keys for my format in experimtens/mapping/multitask
FORMATS = [
    "va",
    "vad",
    "be5",
    "be6",
    "plutchik",
    "izard",
    "pol",
    "pol1",
    "nrc",
    "pol5",
    "be_flickr",
    "be_fer13",
    "be_affectnet",
]


VARIABLES = {
    "va": VA,
    "vad": VAD,
    "be5": BE5,
    "be6": BE6,
    "plutchik": PLUTCHIK,
    "izard": IZARD,
    "pol": POL,
    "pol1": POL1,
    "nrc": NRC,
    "pol5": POL5,
    "be_flickr": BE_FLICKR,
    "be_fer13": BE_FER13,
    "be_affectnet": BE_AFFECTNET
}

# scaling ranges
TANH = "tanh"
SIGMOID = "sigmoid"
SOFTMAX = "softmax" # currently not in use
LOGITS = "logits" # no scaling at all. Used for Multiclass datasets.


SCALINGS = {
    "va": TANH,
    "vad": TANH,
    "be5": SIGMOID,
    "be6": SIGMOID,
}

# prediction problems
MULTIVARIATE_REGRESSION = "regression"
MULTILABEL = "multilabel"
MULTICLASS = "mulitclass"
BINARY = "binary"

FORMATS_2_LOSSES = {
    "va": torch.nn.MSELoss,
    "vad": torch.nn.MSELoss,
    "be5": torch.nn.MSELoss,
    "be6": torch.nn.MSELoss,
    "plutchik": torch.nn.BCEWithLogitsLoss,
    "izard": torch.nn.CrossEntropyLoss,
    "pol": torch.nn.BCEWithLogitsLoss,
    "nrc": torch.nn.BCEWithLogitsLoss,
    "pol5": torch.nn.CrossEntropyLoss,
    "be_flickr": torch.nn.CrossEntropyLoss,
    "be_fer13": torch.nn.CrossEntropyLoss,
    "be_affectnet": torch.nn.CrossEntropyLoss
}


#### ALTERNATIVE VERSION WHERE EVERYTHING IS MODELED EXPLICITLY
# TODO Introduce these changes
#
#
# class Prediction_Problem:
#
#     def __init__(self, name:str, loss_cls:torch.nn.Module.__class__):
#         self.name = name
#         self.loss_cls = loss_cls
#
#
# BINARY_CLASSIFICATION = Prediction_Problem("binary classification", torch.nn.BCEWithLogitsLoss)
# REGRESSION = Prediction_Problem("regression", torch.nn.MSELoss)
# MULTILABEL_CLASSIFICATION = Prediction_Problem("multi-class-multi-label classification", torch.nn.BCEWithLogitsLoss)
# MULTICLASS_CLASSIFICATION = Prediction_Problem("mulit-class-single-label classification", torch.nn.CrossEntropyLoss)
#
#
#
# class Emotion_Variable:
#
#     def __init__(self, *names):
#         self.names = names
#         self.preferred_name = names[0]
#         self.alternative_names = names[1:]
#
# VALENCE = Emotion_Variable("valence", "polarity")
# AROUSAL = Emotion_Variable("arousal")
# DOMINANCE = Emotion_Variable("dominance", "control")
# JOY = Emotion_Variable("joy","happy", "happiness", "joyful")
# ANGER = Emotion_Variable("anger")
# SADNESS = Emotion_Variable("sadness", "sad")
# FEAR = Emotion_Variable("fear", "fearful", "afraid")
# DISGUST = Emotion_Variable("disgust", "disgusted")
# SURPRISE = Emotion_Variable("surprise", "surprised")
# NEUTRAL = Emotion_Variable("neutral", "none")
# ANTICIPATION = Emotion_Variable("anticipation")
# TRUST = Emotion_Variable("trust")
# CONTEMPT = Emotion_Variable("contempt") # "Verachtung"
# AWE = Emotion_Variable("awe")
# AMUSEMENT = Emotion_Variable("amusement") # == joy?
# CONTENTMENT = Emotion_Variable("contentment", "content") # "zufriedenheit"
# GULIT = Emotion_Variable("guilt")
# SHAME = Emotion_Variable("shame")
# POSITIVE = Emotion_Variable("positive")
# NEGATIVE = Emotion_Variable("negative")
#
#
# class Label_Format:
#
#     def __init__(self,
#                  name:str,
#                  variables:list,
#                  scaling:str,
#                  prediction_problem:Prediction_Problem):
#         self.name = name
#         self.variables = variables
#         self.scaling = scaling
#         self.prediction_problem = prediction_problem
#         self.loss_cls = prediction_problem.loss_cls
#
#
# VA = Label_Format(name="va",
#                   variables=[VALENCE, AROUSAL],
#                   scaling=TANH,
#                   prediction_problem=REGRESSION)
#
# VAD = Label_Format(name="vad",
#                    variables=[VALENCE,AROUSAL,DOMINANCE],
#                    scaling=TANH,
#                    prediction_problem=REGRESSION)
#
# BE5 = Label_Format(name="be5",
#                    variables=[JOY, ANGER, SADNESS, FEAR, DISGUST],
#                    scaling=SIGMOID,
#                    prediction_problem=REGRESSION)
#
# BE6 = Label_Format(name="be6",
#                    variables=[JOY, ANGER, SADNESS, FEAR, DISGUST, SURPRISE],
#                    scaling=SIGMOID,
#                    prediction_problem=REGRESSION)
#
# POL1 = Label_Format(name="pol",
#                     variables=[VALENCE],
#                     scaling=LOGITS,
#                     prediction_problem=BINARY_CLASSIFICATION)
#
#
# POL2 = Label_Format(name="pol",
#                     variables=[POSITIVE, NEGATIVE],
#                     scaling=LOGITS,
#                     prediction_problem=BINARY_CLASSIFICATION)
#
# PLUTCHIK = Label_Format(name="plutchik",
#                    variables=[JOY, ANGER, SADNESS, FEAR, DISGUST, SURPRISE, DISGUST, SURPRISE, ANTICIPATION, TRUST],
#                    scaling=SIGMOID,
#                    prediction_problem=REGRESSION)
#
# NRC = Label_Format(name="nrc",
#                    variables=[JOY, ANGER, SADNESS, FEAR, DISGUST, SURPRISE, DISGUST, SURPRISE, ANTICIPATION, TRUST,
#                               POSITIVE, NEGATIVE],
#                    scaling=SIGMOID,
#                    prediction_problem=REGRESSION)
#
# BE_FLICKR = Label_Format("be_flickr",
#                          variables=[AMUSEMENT, ANGER, AWE, "contentment", "disgust", "excitement", "fear", "sadness"],
#                          scaling=LOGITS,
#                          prediction_problem=MULTICLASS_CLASSIFICATION)
#
# BE_FER13 = Label_Format("be_fer13",
#                          variables=[ANGER, DISGUST, FEAR, JOY, SADNESS, SURPRISE, NEUTRAL],
#                          scaling=LOGITS,
#                          prediction_problem=MULTICLASS_CLASSIFICATION)
#
# BE_AFFECTNET = Label_Format("be_fer13",
#                          variables=["neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger", "contempt"],
#                          scaling=LOGITS,
#                          prediction_problem=MULTICLASS_CLASSIFICATION)
#
#
#


