
import os
import argparse
import random
import numpy as np
import torch

from chem_lib.datasets import obatin_train_test_tasks
from chem_lib.utils import init_trial_path

def get_parser(root_dir='.'):
    parser = argparse.ArgumentParser(description='Property-Aware Relation Networks for Few-Shot Molecular Property Prediction')
    # dataset
    parser.add_argument('-r', '--root-dir', type=str, default=root_dir, help='root-dir')
    parser.add_argument('-d', '--dataset', type=str, default='sider', help='data set name')  # ['tox21','sider','muv']
    parser.add_argument('-td', '--test-dataset', type=str, default='sider',
                        help='test data set name')  # ['tox21','sider','muv']
    parser.add_argument('--data-dir', type=str, default=os.path.join(root_dir,'data') +'/', help='data dir')
    parser.add_argument('--preload_train_data', type=bool, default=True)  # 0
    parser.add_argument('--preload_test_data', type=bool, default=True)  # 0
    parser.add_argument("--run_task", type=int, default=-1, help="run on task")
    # few shot
    parser.add_argument("--n-shot-train", type=int, default=10, help="train: number of shot for each class")
    parser.add_argument("--n-shot-test", type=int, default=10, help="test: number of shot for each class")
    parser.add_argument("--n-query", type=int, default=16, help="number of query in few shot learning")
    # training
    parser.add_argument("--meta-lr", type=float, default=0.001,
                        help="Training: Meta learning rate")  
    parser.add_argument("--weight_decay", type=float, default=5e-5,
                        help="Training: Meta learning weight_decay")  
    parser.add_argument("--inner-lr", type=float, default=0.1, help="Training: Inner loop learning rate")
    parser.add_argument('--epochs', type=int, default=6000,help='number of epochs to train ')   #6000
    parser.add_argument('--update_step', type=int, default=1)
    parser.add_argument('--update_step_test', type=int, default=1)
    parser.add_argument('--inner_update_step', type=int, default=1)
    parser.add_argument("--meta_warm_step", type=int, default=0, help="meta warp up step for encode")
    parser.add_argument("--meta_warm_step2", type=int, default=10000, help="meta warp up step 2 for encode")
    parser.add_argument("--second_order", type=int, default=1, help="second order or not")
    parser.add_argument("--batch_task", type=int, default=9, help="Training: Meta batch size")
    parser.add_argument("--adapt_weight", type=int, default=5, help="adaptable weights")
    parser.add_argument("--eval_support", type=int, default=0, help="Training: eval s")
    # model
    ## mol-encoder
    parser.add_argument('--enc_gnn', type=str, default="gin") #choices=["gin", "gcn", "gat", "graphsage"]
    parser.add_argument('--enc_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')#last
    parser.add_argument('--enc_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--enc_batch_norm', type=int, default=1,
                        help='use batch norm or not')
    parser.add_argument('--pretrained', type=int, default=1, help='pretrained or not')
    parser.add_argument('--pretrained_weight_path', type=str,
                        default=os.path.join(root_dir,'chem_lib/model_gin/supervised_contextpred.pth'), help='pretrained path')

    parser.add_argument("--rel_dropout2", type=float, default=0.2, help="rel dropout2")
    # other
    parser.add_argument('--seed', type=int, default=11, help="Seed for splitting the dataset.")
    parser.add_argument('--gpu_id', type=int, default=5, help="Choose the number of GPU.")
    parser.add_argument("--result_path", type=str, default=os.path.join(root_dir,'results_tsne'), help="result path")
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=2000, help="Training: Number of iterations between checkpoints")
    parser.add_argument("--save-logs", type=int, default=0)
    parser.add_argument("--support_valid", type=int, default=0)
    # parser.add_argument("--top_k", type=int, default=5, help="top k")## 3 for 5-shot,5 for 10-shot
    return parser



def get_args(root_dir='.',is_save=True):
    parser = get_parser(root_dir)
    args = parser.parse_args()

    args.rel_k= args.n_shot_train
    if args.n_shot_train==10:
        args.top_k = 5
    elif args.n_shot_train==5:
        args.top_k = 3
    if args.pretrained:
        args.enc_layer = 5
        args.emb_dim =300
        args.dropout = 0.5 
    if  args.enc_layer <= 3:
        args.emb_dim =200
        args.dropout = 0.1

    if args.test_dataset == args.dataset:
        args.test_dataset = None
    # if args.map_layer<=0:
    #     args.map_dim = args.emb_dim
    args = init_trial_path(args,is_save)
    device = "cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu"
    args.device = device
    print(args)

    train_tasks, test_tasks = obatin_train_test_tasks(args.dataset)
    if args.test_dataset is not None:
        train_tasks_2, test_tasks_2 = obatin_train_test_tasks(args.test_dataset)
        train_tasks = train_tasks + test_tasks
        test_tasks = train_tasks_2 + test_tasks_2
    train_tasks=sorted(list(set(train_tasks)))
    test_tasks=sorted(list(set(test_tasks)))
    if args.run_task>=0:
        train_tasks=[args.run_task]
        test_tasks=[args.run_task]
    args.train_tasks = train_tasks
    args.test_tasks = test_tasks

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    return args