import argparse
import os
import json
import pandas as pd
parser = argparse.ArgumentParser()



parser.add_argument('--lr', type=float, default=0.001) 
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--data_path', type=str, default='../yelp/seq_yelp.csv')
parser.add_argument('--save_path', type=str, default='save/')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_len', type=int, default=20) 
parser.add_argument('--train_batch_size', type=int, default=128) 
parser.add_argument('--test_batch_size', type=int, default=128)



parser.add_argument('--d_model', type=int, default=128) 
parser.add_argument('--enable_res_parameter', type=int, default=1) 

parser.add_argument('--attn_heads', type=int, default=4) 
parser.add_argument('--dropout', type=float, default=0.1) 
parser.add_argument('--alpha', type=float, default=-1)
parser.add_argument('--d_ffn', type=int, default=256)  
parser.add_argument('--bert_layers', type=int, default=2) 

parser.add_argument('--baserate', type=float, default=1)
parser.add_argument('--t', type=float, default=5)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--ld1', type=float, default=1e-3)
parser.add_argument('--ld2', type=float, default=1e-3)
parser.add_argument('--neg_samples', type=int, default=10000)
parser.add_argument('--seed', type=int, default=0)




args = parser.parse_args()


DATA = pd.read_csv(args.data_path, header=None)
args.num_user = len(DATA)
num_item = DATA.max().max()
del DATA
args.num_item = int(num_item)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
