import pandas as pd
import torch
from torch import nn
import torch.utils.data as Data
from tqdm import tqdm
from dataset import TrainDataset,EvalDataset,TestDataset
import warnings 
warnings.filterwarnings('ignore') 
from sasrec import SASRec
import random
import pickle
import numpy as np



def neg_sample(seq, labels, num_item,sample_size):
    negs = set()
    seen = set(labels)

    while len(negs) < sample_size:
        candidate = np.random.randint(0, num_item) + 1
        while candidate in seen or candidate in negs:
            candidate = np.random.randint(0, num_item) + 1
        negs.add(candidate)
    return negs


def recalls_and_ndcgs_for_ks(scores, labels, ks): 
        metrics = {}
        answer_count = labels.sum(1)
        labels_float = labels.float()
        rank = (-scores).argsort(dim=1)
        cut = rank
        for k in sorted(ks, reverse=True):
            cut = cut[:, :k]
            hits = labels_float.gather(1, cut)

            metrics['Recall@%d' % k] = \
                (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device),
                                         labels.sum(1).float())).mean().cpu().item()

            position = torch.arange(2, 2 + k)
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights.to(hits.device)).sum(1)
            idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
            ndcg = (dcg / idcg).mean()
            metrics['NDCG@%d' % k] = ndcg.cpu().item()

            v,p = torch.max(hits,dim=-1)
            mrr = v / (p.float()+1) 
            mrr = mrr.mean()
            metrics['MRR@%d' % k] = mrr.cpu().item()
        return metrics

def eval(data_loader,model,metric_ks,num_item,device):
    
    model.eval()
    with torch.no_grad():
        
        metrics = {}

        for idx,(seq,label) in enumerate(tqdm(data_loader)):
            #L1 = torch.zeros(512,70000).cuda()
            seq = seq.to(device)
            label = label.to(device)
            
            len_ = (seq.sum(dim=0) == 0).sum()
            position = torch.arange(seq.shape[1],device=device).unsqueeze(0).repeat(len(seq),1) 
            x = model.inf(seq[:,len_:],position[:,len_:])


            
            answers = label.tolist()
            seqs = seq.tolist()
            label = label.view(-1)
            #labels = label.view(-1)#.tolist()

            row = []
            col = []
            for i in range(len(answers)):
                seq = list(set(seqs[i]))
                #seq.remove(answers[i][0])
                row += [i] * len(seq)
                col += seq
            x[row, col] = -1e9
            #labels = torch.nn.functional.one_hot(torch.tensor(labels, dtype=torch.int64,device=device), num_classes=num_item+1)
            labels = torch.zeros(len(label),num_item+1,dtype=torch.int64,device=device)
            index_x = torch.arange(len(label),device=device)
            labels[index_x,label] = 1

            metrics_batch = recalls_and_ndcgs_for_ks(x,labels, metric_ks)
            for k, v in metrics_batch.items():
                if not metrics.__contains__(k):
                    metrics[k] = v
                else:
                    metrics[k] += v
            #L1[L1==1] = 0
        for k, v in metrics.items():
            metrics[k] = v/(idx+1)
    model.train()
    return metrics 


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic= False
    torch.cuda.manual_seed_all(seed)

def main(args):
    seed_everything(args.seed)
    stop = 0
    device = args.device
    args.eval_per_steps = args.num_user // args.train_batch_size #// 1000
    train_dataset = TrainDataset(args)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    eval_dataset = EvalDataset(args)
    eval_loader = Data.DataLoader(eval_dataset, batch_size=args.test_batch_size)

    test_dataset = TestDataset(args)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    print('dataset initial ends')
    model = SASRec(args).to(args.device)

    CE = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)


    step = best = 0
    baserate = args.baserate

    for epoch in range(args.num_epoch):
        model.train()
       
        total = total_ritem = total_rtarget = total_dp =  0
        pbar = tqdm(train_loader)
        for idx,(seq,label) in (enumerate(pbar)):
            tau = args.t
            step += 1
            seq = seq.to(args.device)
            label = label.to(args.device)
            # neg
            negs = neg_sample([],[],args.num_item,10000)
            negs = torch.tensor(list(negs),device=device,dtype=torch.int64).to(args.device)
            # prob

            position = torch.arange(args.max_len,device=device).unsqueeze(0).repeat(len(seq),1)    
            len_ = (seq.sum(dim=1) == 0).sum()
            out = model(seq[:,len_:],label[:,len_:],negs,position = position[:,len_:])
            loss = CE(out,torch.LongTensor([0]*out.size(0)).to(args.device))

            opt.zero_grad()
            loss.backward()
            opt.step()

            
            with torch.no_grad():
                total += loss.cpu().item()
                pbar.set_postfix({'L':total/(idx+1)})
                    
          


            if step % args.eval_per_steps == 0:
                model.eval()

                with torch.no_grad():
                    
                    m = eval(eval_loader,model,[20,10],args.num_item,args.device) 
                    print('validation',epoch,m)
                    f = open(args.save_path + 'result.txt','a+')
                    print(epoch,m,file = f)
                    if m['NDCG@10'] > best:
                        g = open(args.save_path +'model'+str(args.t)+str(baserate)+'.pkl','wb')
                        pickle.dump(model,g)
                        g.close()
                        best = m['NDCG@10']
                        stop = 0
                    else:
                        stop += 1
                        if stop == 3: return
                    f.close()
                model.train()

        f = open(args.save_path + 'loss.txt','a+')
        print('epoch',epoch,'loss',total/(idx+1),file=f)
        f.close()

if __name__ == '__main__':
    from args import args
    main(args)
