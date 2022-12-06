import subprocess
from emocoder.experiments import constants
from emocoder.src import utils
import argparse


def main(split):
    
    aggregation_script_path = utils.get_script_dir() / "aggregate_experiment_group.py"



    ### MAPPING

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.MAPPING_BASELINE_BASEPATH / split,
                    "--removetimestamp"])

    # TODO Warum werden nur bei den Multitask experimenten (auch bei word und text) Fallunterscheidungen für den Split gebraucht?
    if split == "dev":
        subprocess.run(["python",
                        aggregation_script_path,
                        constants.MAPPING_MULTITASK_BASEPATH / split,
                        "--removetimestamp",
                        "--multitask"])
    elif split == "test":
        subprocess.run(["python",
                        aggregation_script_path,
                        constants.MAPPING_MULTITASK_BASEPATH / split,
                        "--removetimestamp"])


    ### WORD

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.WORD_BASELINE_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.WORD_PROPOSED_BASEPATH / split,
                     "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.WORD_AUGMENTED_BASEPATH / split,
                    "--removetimestamp"])

    if split == "dev":
        subprocess.run(["python",
                        aggregation_script_path,
                        constants.WORD_MULTITASK_BASEPATH / split,
                        "--removetimestamp",
                        "--multitask"])
    elif split == "test":
        subprocess.run(["python",
                        aggregation_script_path,
                        constants.WORD_MULTITASK_BASEPATH / split,
                        "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.WORD_ZEROSHOTBASELINE_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.WORD_ZEROSHOT_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.WORD_AUGZERO_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.WORD_AUGMENTED_BASEPATH / split,
                    "--removetimestamp"])


    ### TEXT

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.TEXT_BASELINE_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.TEXT_PROPOSED_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.TEXT_AUGMENTED_BASEPATH / split,
                    "--removetimestamp"])


    if split == "dev":
        subprocess.run(["python",
                        aggregation_script_path,
                        constants.TEXT_MULTITASK_BASEPATH / split,
                        "--removetimestamp",
                        "--multitask"])
    elif split == "test":
        subprocess.run(["python",
                        aggregation_script_path,
                        constants.TEXT_MULTITASK_BASEPATH / split,
                        "--removetimestamp"
                        ])


    subprocess.run(["python",
                    aggregation_script_path,
                    constants.TEXT_ZEROSHOTBASELINE_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.TEXT_ZEROSHOT_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.TEXT_AUGZERO_BASEPATH / split,
                    "--removetimestamp"])


    ### Image

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.IMAGE_BASELINE_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.IMAGE_PROPOSED_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.IMAGE_AUGMENTED_BASEPATH / split,
                    "--removetimestamp"])


    if split == "dev":
        subprocess.run(["python",
                        aggregation_script_path,
                        constants.IMAGE_MULTITASK_BASEPATH / split,
                        "--removetimestamp",
                        "--multitask"])
    elif split == "test":
        subprocess.run(["python",
                        aggregation_script_path,
                        constants.IMAGE_MULTITASK_BASEPATH / split,
                        "--removetimestamp"
                        ])


    subprocess.run(["python",
                    aggregation_script_path,
                    constants.IMAGE_ZEROSHOTBASELINE_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.IMAGE_ZEROSHOT_BASEPATH / split,
                    "--removetimestamp"])

    subprocess.run(["python",
                    aggregation_script_path,
                    constants.IMAGE_AUGZERO_BASEPATH / split,
                    "--removetimestamp"])







if __name__== "__main__":
    parser = argparse.ArgumentParser("Aggregates experimental results for all experiment groups")
    parser.add_argument("--split", default="dev", help="Whether to aggregate all 'dev' or all 'test' experiments.")
    args = parser.parse_args()
    assert args.split in ["dev", "test"]
    main(split=args.split)