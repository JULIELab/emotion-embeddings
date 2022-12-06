"""
This works like a registry for experimental results.
"""

from emocoder.src import utils



# ===========================   MAPPING  ===================================== #
MAPPING_BASELINE_BASEPATH = utils.get_target_dir() / "mapping" / "baseline"
MAPPING_PROPOSED_BASEPATH = utils.get_target_dir() / "mapping" / "proposed"
MAPPING_MULTITASK_BASEPATH = utils.get_target_dir() / "mapping" / "multitask"




# ===========================   WORD  ===================================== #
WORD_BASELINE_BASEPATH = utils.get_target_dir() / "word"/ "baseline"
WORD_PROPOSED_BASEPATH = utils.get_target_dir() / "word" / "proposed"
WORD_MULTITASK_BASEPATH = utils.get_target_dir() / "word" / "multitask"
WORD_AUGMENTED_BASEPATH = utils.get_target_dir() / "word" / "augmented"
WORD_ZEROSHOT_BASEPATH = utils.get_target_dir() / "word" / "zeroshot"
WORD_AUGZERO_BASEPATH = utils.get_target_dir() / "word" / "augzero"
WORD_ZEROSHOTBASELINE_BASEPATH = utils.get_target_dir() / "word" / "zeroshotbaseline"




# ===========================   TEXT  ===================================== #
TEXT_BASELINE_BASEPATH = utils.get_target_dir() / "text" / "baseline"
TEXT_PROPOSED_BASEPATH  = utils.get_target_dir() / "text" / "proposed"
TEXT_MULTITASK_BASEPATH = utils.get_target_dir() / "text" / "multitask"
TEXT_AUGMENTED_BASEPATH = utils.get_target_dir() / "text" / "augmented"
TEXT_ZEROSHOT_BASEPATH = utils.get_target_dir() / "text" / "zeroshot"
TEXT_AUGZERO_BASEPATH = utils.get_target_dir() / "text" / "augzero"
TEXT_ZEROSHOTBASELINE_BASEPATH = utils.get_target_dir() / "text" / "zeroshotbaseline"


# ===========================   IMAGES  ===================================== #
IMAGE_BASELINE_BASEPATH = utils.get_target_dir() / "image" / "baseline"
IMAGE_PROPOSED_BASEPATH  = utils.get_target_dir() / "image" / "proposed"
IMAGE_MULTITASK_BASEPATH = utils.get_target_dir() / "image" / "multitask"
IMAGE_AUGMENTED_BASEPATH = utils.get_target_dir() / "image" / "augmented"
IMAGE_ZEROSHOT_BASEPATH = utils.get_target_dir() / "image" / "zeroshot"
IMAGE_AUGZERO_BASEPATH = utils.get_target_dir() / "image" / "augzero"
IMAGE_ZEROSHOTBASELINE_BASEPATH = utils.get_target_dir() / "image" / "zeroshotbaseline"




EXPERIMENT_DIRS = [
    MAPPING_BASELINE_BASEPATH,
    MAPPING_PROPOSED_BASEPATH,
    MAPPING_MULTITASK_BASEPATH,
    WORD_BASELINE_BASEPATH,
    WORD_PROPOSED_BASEPATH,
    WORD_AUGMENTED_BASEPATH,
    WORD_MULTITASK_BASEPATH,
    WORD_ZEROSHOTBASELINE_BASEPATH,
    WORD_ZEROSHOT_BASEPATH,
    WORD_AUGZERO_BASEPATH,
    TEXT_BASELINE_BASEPATH,
    TEXT_PROPOSED_BASEPATH,
    TEXT_AUGMENTED_BASEPATH,
    TEXT_MULTITASK_BASEPATH,
    TEXT_ZEROSHOTBASELINE_BASEPATH,
    TEXT_ZEROSHOT_BASEPATH,
    TEXT_AUGZERO_BASEPATH,
    IMAGE_BASELINE_BASEPATH,
    IMAGE_PROPOSED_BASEPATH,
    IMAGE_AUGMENTED_BASEPATH,
    IMAGE_MULTITASK_BASEPATH,
    IMAGE_ZEROSHOTBASELINE_BASEPATH,
    IMAGE_ZEROSHOT_BASEPATH,
    IMAGE_AUGZERO_BASEPATH,
]


