import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorflow.python.keras.saving.save import save_model
from torchmetrics.functional import auroc
# from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
# from sklearn.metrics import auc as PR_AUC


from torch_geometric.data import DataLoader

from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Meta_Trainer(nn.Module):
    def __init__(self, args, model):
        super(Meta_Trainer, self).__init__()
        self.args = args
        self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query
        self.device = args.device
        self.emb_dim = args.emb_dim
        self.batch_task = args.batch_task
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step
        self.top_k=args.top_k
        self.trial_path = args.trial_path

        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t + 1) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid', 'AUC-Best']
        logger.set_names(log_names)
        self.logger = logger
        preload_train_data = {}
        if args.preload_train_data:
            print('preload train data')
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/ll/" + str(task+1),
                                          dataset=self.dataset)
                print(self.data_dir + self.dataset + "/ll/"+str(task+1))

                preload_train_data[task] = dataset
        preload_test_data = {}
        if args.preload_test_data:
            print('preload_test_data')
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/ll/" + str(task+1),
                                          dataset=self.test_dataset)
                print(self.data_dir + self.dataset + "/ll/"+str(task+1))

                preload_test_data[task] = dataset
        self.preload_train_data = preload_train_data
        self.preload_test_data = preload_test_data
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train','valid')
            print('preload_valid_data')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/ll/" + str(task+1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc=0


        self.res_logs=[]

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        for samples in loader:
            samples=samples.to(self.device)
            return samples

    def get_data_sample(self, task_id, train=True):
        if train:
            task = self.train_tasks[task_id]
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/ll/" + str(task+1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)

            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                            'label': torch.cat([s_data.y, q_data.y], 0)}
            eval_data = { }
        else:
            task = self.test_tasks[task_id]
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/ll/" + str(task+1),
                                          dataset=self.test_dataset)
            s_data, q_data,q_adapt= sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)
            q_data_adapt = self.loader_to_samples(q_adapt)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data_adapt, 'q_label': q_data_adapt.y}

            eval_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y }
        return adapt_data, eval_data

    def get_prediction(self, model, data,train=True):
        s_logits, q_logits= model(data['s_data'], data['q_data'])
        pred_dict = {'s_logits': s_logits, 'q_logits': q_logits}
        return pred_dict

    def get_adaptable_weights(self, model, adapt_weight=None):
        if adapt_weight is None:
            adapt_weight = self.args.adapt_weight
        fenc = lambda x: x[0]== 'mol_encoder'
        frel = lambda x: x[0]== 'adapt_relation'
        fedge = lambda x: x[0]== 'adapt_relation' and 'edge_layer'  in x[1]
        fnode = lambda x: x[0]== 'adapt_relation' and 'node_layer'  in x[1]
        fclf = lambda x: x[0]== 'adapt_relation' and 'fc'  in x[1]
        if adapt_weight==0:
            flag=lambda x: not fenc(x)
        elif adapt_weight==1:
            flag=lambda x: not frel(x)
        elif adapt_weight==2:
            flag=lambda x: not (fenc(x) or frel(x))
        elif adapt_weight==3:
            flag=lambda x: not (fenc(x) or fedge(x))
        elif adapt_weight==4:
            flag=lambda x: not (fenc(x) or fnode(x))
        elif adapt_weight==5:
            flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
        elif adapt_weight==6:
            flag=lambda x: not (fenc(x) or fclf(x))
        else:
            flag= lambda x: True
        if self.train_epoch < self.args.meta_warm_step or self.train_epoch>self.args.meta_warm_step2:
            adaptable_weights = None
        else:
            adaptable_weights = []
            adaptable_names=[]
            for name, p in model.module.named_parameters():
                names=name.split('.')
                if flag(names):
                    adaptable_weights.append(p)
                    adaptable_names.append(name)
        return adaptable_weights

    def get_loss(self, model, batch_data, pred_dict,flag=0):
        """
        flag = 0 -> query
        flag = 1 -> support
        """
        if flag == 0:
            loss = self.criterion(pred_dict['q_logits'], batch_data['q_label'])
        else:
            loss = self.criterion(pred_dict['s_logits'], batch_data['s_label'])
        return loss
    def train_step(self):
        self.train_epoch += 1
        task_id_list = list(range(len(self.train_tasks)))
        # if self.batch_task > 0:
        #     batch_task = min(self.batch_task, len(task_id_list))
        #     task_id_list = random.sample(task_id_list, batch_task)
        data_batches = {}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id] = db

        for k in range(self.update_step):
            losses_eval = []
            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]
                model = self.model.clone()
                model.train()
                adaptable_weights = self.get_adaptable_weights(model)
                for inner_step in range(self.inner_update_step):
                    pred_adapt = self.get_prediction(model, train_data, train=True)
                    loss_adapt = self.get_loss(model, train_data, pred_adapt, flag=1)
                    model.adapt(loss_adapt, adaptable_weights=adaptable_weights)
                pred_eval = self.get_prediction(model, train_data, train=True)
                loss_eval = self.get_loss(model, train_data, pred_eval, flag=0)
                losses_eval.append(loss_eval)
            losses_eval = torch.stack(losses_eval)
            losses_eval = torch.sum(losses_eval)
            losses_eval = losses_eval / len(task_id_list)
            self.optimizer.zero_grad()
            losses_eval.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            print('Train Epoch:', self.train_epoch, ', train update step:', k, ', loss_eval:', losses_eval.item())
        return self.model.module
    def test_step(self):
        step_results={'query_preds':[], 'query_labels':[], 'query_adj':[],'task_index':[]}
        auc_scores = []
        for task_id in range(len(self.test_tasks)):
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)
            model = self.model.clone()
            model.train()
            adaptable_weights = self.get_adaptable_weights(model)
            pred_adapt = self.get_prediction(model, adapt_data, train=True)
            loss_adapt = self.get_loss(model, adapt_data, pred_adapt, flag=1)
            model.adapt(loss_adapt, adaptable_weights=adaptable_weights)
            model.eval()
            with torch.no_grad():

                pred_eval = self.get_prediction(model, eval_data, train=False)
                y_score = F.softmax(pred_eval['q_logits'], dim=-1).detach()[:, 1]
                y_true = eval_data['q_label']
                auc = auroc(y_score,y_true, task="binary").item()
            auc_scores.append(auc)
            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4) )
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.cpu().numpy())
                step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])
        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc,avg_auc)
        self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)
        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),', Best_Avg_AUC: ', round(self.best_auc, 4),)
        if self.args.save_logs:
            self.res_logs.append(step_results)
        return self.best_auc

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        torch.save(self.model.module.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)