import argparse
from utils.data_loader import load_data
from train import train_test

parser = argparse.ArgumentParser()
parser.add_argument('--rootpath', type=str, default='../../data/', help='datapath')
parser.add_argument('--task', type=str, default='recommendation', help='others: category_classification local_news_detection popularity_prediction')

##turning paras
parser.add_argument('--learning_rate', action='store_true', default=0.005, help='learning rate')
parser.add_argument('--epoch', action='store_true', default=100, help='epoch num')
parser.add_argument('--batch_size', action='store_true', default=500, help='batch size')
parser.add_argument('--l2_regular', action='store_true', default=0.00001, help='l2 regular')

parser.add_argument('--news_entity_num', action='store_true', default=20, help='fix a news entity num to news_entity_num')
parser.add_argument('--negative_num', action='store_true', default=6, help='negative sampling number')
parser.add_argument('--embedding_dim', action='store_true', default=90, help='embedding dim for enity_embedding dv uv')
parser.add_argument('--layer_dim', action='store_true', default=128, help='layer dim')


args = parser.parse_args()

data = load_data(args)

train_test(args, data)
