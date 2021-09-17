import argparse

from FairNeuron import Fixate_with_val



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices={'compas','census','credit'})
    parser.add_argument('--epoch')
    parser.add_argument('--save-dir',default='./results')
    parser.add_argument('--rand',action='store_true')
    args=parser.parse_args()

    Fixate_with_val(epoch=args.epoch,dataset=args.dataset)
