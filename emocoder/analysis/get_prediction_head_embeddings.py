"""
Stores the weight vectors of the prediction heads in a csv file.
"""

import pandas as pd
import seaborn as sns
sns.set(style="white")
from emocoder.experiments import utils as xutils
from emocoder.src.utils import get_analysis_dir

model, state_dict = xutils.get_pretrained_emotion_codec()

vectors = {}


### Version where vad/be5 decoders have no activation
vectors["val_vad"] = state_dict.get("dec.pool.vad.weight")[0]
vectors["aro_vad"] = state_dict.get("dec.pool.vad.weight")[1]
vectors["dom_vad"] = state_dict.get("dec.pool.vad.weight")[2]

vectors["val_va"] = state_dict.get("dec.pool.va.weight")[0]
vectors["aro_va"] = state_dict.get("dec.pool.va.weight")[1]

vectors["joy_be5"] = state_dict.get("dec.pool.be5.weight")[0]
vectors["ang_be5"]  =state_dict.get("dec.pool.be5.weight")[1]
vectors["sad_be5"] =state_dict.get("dec.pool.be5.weight")[2]
vectors["fea_be5"] =state_dict.get("dec.pool.be5.weight")[3]
vectors["dis_be5"] =state_dict.get("dec.pool.be5.weight")[4]

### Version where vad/be5 decoders have sigmoid activation
# vectors["val"] = state_dict.get("dec.pool.vad.linear.weight")[0]
# vectors["aro"] = state_dict.get("dec.pool.vad.linear.weight")[1]
# vectors["dom"] = state_dict.get("dec.pool.vad.linear.weight")[2]
#
# vectors["joy_be5"] = state_dict.get("dec.pool.be5.linear.weight")[0]
# vectors["ang_be5"]  =state_dict.get("dec.pool.be5.linear.weight")[1]
# vectors["sad_be5"] =state_dict.get("dec.pool.be5.linear.weight")[2]
# vectors["fea_be5"] =state_dict.get("dec.pool.be5.linear.weight")[3]
# vectors["dis_be5"] =state_dict.get("dec.pool.be5.linear.weight")[4]

### END CASE DISTINCTION

vectors["anger_fer"] = state_dict.get("dec.pool.be_fer13.weight")[0]
vectors["disgust_fer"] = state_dict.get("dec.pool.be_fer13.weight")[1]
vectors["fear_fer"] = state_dict.get("dec.pool.be_fer13.weight")[2]
vectors["happy_fer"] = state_dict.get("dec.pool.be_fer13.weight")[3]
vectors["sad_fer"] = state_dict.get("dec.pool.be_fer13.weight")[4]
vectors["surprise_fer"] = state_dict.get("dec.pool.be_fer13.weight")[5]
vectors["neutral_fer"] = state_dict.get("dec.pool.be_fer13.weight")[6]

vectors["neutral_an"] = state_dict.get("dec.pool.be_affectnet.weight")[0]
vectors["happy_an"] = state_dict.get("dec.pool.be_affectnet.weight")[1]
vectors["sad_an"] = state_dict.get("dec.pool.be_affectnet.weight")[2]
vectors["surprise_an"] = state_dict.get("dec.pool.be_affectnet.weight")[3]
vectors["fear_an"] = state_dict.get("dec.pool.be_affectnet.weight")[4]
vectors["disgust_an"] = state_dict.get("dec.pool.be_affectnet.weight")[5]
vectors["anger_an"] = state_dict.get("dec.pool.be_affectnet.weight")[6]
vectors["contempt_an"] = state_dict.get("dec.pool.be_affectnet.weight")[7]


vectors = {key: val.numpy() for key,val in vectors.items()}
variables = pd.DataFrame.from_dict(vectors, orient="index")

variables.to_csv(get_analysis_dir() / "prediction_head_embeddings.csv")