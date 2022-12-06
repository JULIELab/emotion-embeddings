import argparse

def main(gpu=0):
    # have to import one after the other, because experiments are dependent form each others output
    from emocoder.experiments.word import proposed, augmented, multitask
    proposed.run_all_dev_exp(gpu)
    augmented.run_all_dev_exp(gpu)
    multitask.run_all_dev_exp(gpu)
    from emocoder.experiments.word import zeroshot, augzero
    zeroshot.run_all_dev_exp(gpu)
    augzero.run_all_dev_exp(gpu)


    from emocoder.experiments.text import proposed, augmented, multitask
    proposed.run_all_dev_exp(gpu)
    augmented.run_all_dev_exp(gpu)
    multitask.run_all_dev_exp(gpu)
    from emocoder.experiments.text import zeroshot, augzero
    zeroshot.run_all_dev_exp(gpu)
    augzero.run_all_dev_exp(gpu)

    from emocoder.experiments.image import proposed, augmented, multitask
    proposed.run_all_dev_exp(gpu)
    augmented.run_all_dev_exp(gpu)
    multitask.run_all_dev_exp(gpu)
    from emocoder.experiments.image import zeroshot, augzero
    zeroshot.run_all_dev_exp(gpu)
    augzero.run_all_dev_exp(gpu)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Which GPU to run the experiments on.", type=int, default=0)
    args = parser.parse_args()
    main(gpu=args.gpu)

