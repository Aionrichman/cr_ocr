import os
import argparse

from operate_model import DeepModel
from preprocess_data import get_data_flow


parser = argparse.ArgumentParser(description='')
parser.add_argument('--operation', dest='operation', default='train')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)
parser.add_argument('--epoch', dest='epoch', type=int, default=20000)
parser.add_argument('--last_model', dest='last_model', default='none')
args = parser.parse_args()


if __name__ == '__main__':
    if args.operation == 'train':
        train_dir = "train"
        train_flow = get_data_flow(train_dir, args.batch_size)

        test_dir = "test"
        test_flow = get_data_flow(test_dir, args.batch_size)

        label_num = len(os.listdir(train_dir))

        if args.last_model != "none":
            deep_model = DeepModel(32, 32, label_num, args.last_model)
        else:
            deep_model = DeepModel(32, 32, label_num)
        deep_model.train_model(train_flow, test_flow, args.epoch, './models', './logs')

    elif args.operation == 'test':
        res = DeepModel.test_model(args.last_model, "FZSTK.TTF.png")
        print(res)
