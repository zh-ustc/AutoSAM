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
from sampler import sampler_trm


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



def eval(data_loader,model,sampler,metric_ks,num_item,device,tau):
    model.eval()
    with torch.no_grad():
        metrics = {}
        for idx,(seq,label) in enumerate(tqdm(data_loader)):
            seqs = seq.tolist()
            seq = seq.to(device)
            label = label.to(device)
            e = model.token(seq)+model.position(seq)
            p = sampler(e,seq,tau)
            p[:,-1] = 1
            action = torch.rand(p.shape,device=device) < p
            fill_pos = torch.sort((action).float(),dim=-1)[0]
            seq_new = torch.zeros(seq.shape,device = device,dtype = torch.int64)
            seq_new[fill_pos==1] = seq[action]
            len_ = (seq_new.sum(dim=0) == 0).sum()
            position = torch.arange(seq.shape[1],device=device).unsqueeze(0).repeat(len(seq_new),1)
            position[~action] = 0
            position = torch.sort(position,dim=-1)[0]
            x = model.inf(seq_new[:,len_:],position[:,len_:])
            row = []
            col = []
            for i in range(len(label)):
                seq = list(set(seqs[i]))
                row += [i] * len(seq)
                col += seq
            x[row, col] = -1e9
            labels = torch.nn.functional.one_hot(torch.tensor(label.view(-1), dtype=torch.int64,device=device), num_classes=num_item+1)       
            metrics_batch = recalls_and_ndcgs_for_ks(x,labels, metric_ks)
            for k, v in metrics_batch.items():
                if not metrics.__contains__(k):
                    metrics[k] = v
                else:
                    metrics[k] += v

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
    sampler = sampler_trm(args).to(args.device)
    CE = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt_sampler = torch.optim.SGD(sampler.parameters(), lr=0.1)

    step = best = 0
    baserate = args.baserate

    for epoch in range(args.num_epoch):
        model.train()
        sampler.train()
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
            p = sampler((model.token(seq)+model.position(seq)),seq,tau)
            p = p.masked_fill(seq==0,0)
            action = torch.rand(p.shape,device=device) < p
            action[seq==0] = 0
            # reconstruct seq
            # seqx : [1,2,3,4,5] -> [1,0,3,4,0]
            fill_pos = torch.sort((action).float(),dim=-1)[0]
            seqx = seq.clone().detach()
            seqx[~action] = 0
            # seqrl : [1,2,3,4,5] -> [0,0,1,3,4]
            seqrl = torch.zeros(seq.shape,device = device,dtype = torch.int64)
            seqrl[fill_pos==1] = seq[action]
            position = torch.arange(args.max_len,device=device).unsqueeze(0).repeat(len(seqrl),1)
            position[~action] = 0
            position = torch.sort(position,dim=-1)[0]
            
            # labelrl : [2,3,4,5,6] -> [0,0,3,4,6]
            labelrl = torch.cat([seqrl[:,1:],label[:,-1].unsqueeze(-1)],dim=-1)
            labelrl[seqrl==0] = 0
            # labelrl_train : [2,3,4,5,6] -> [3,0,4,6,0]
            labelrl_train = torch.zeros(labelrl.shape,device=device,dtype=torch.int64)
            labelrl_train[action] = labelrl[seqrl>0]

            len_ = (seqrl.sum(dim=1) == 0).sum()
            out_new = model(seqrl[:,len_:],labelrl[:,len_:],negs,position = position[:,len_:])
            loss = CE(out_new,torch.LongTensor([0]*out_new.size(0)).to(args.device))

            with torch.no_grad():
                
                ''' Reward Pre '''
                    # keep n + 1 step
                out = model(seq,label,negs,position=None)
                out = out.softmax(dim=-1)
                logprob = out[:,0].log().clone().detach()
                
                R_itemb = torch.zeros(action.shape,device=device)
                R_itemb[label>0] = logprob
                R_item = torch.zeros(action.shape,device=device)

                
                x,y = torch.max(labelrl_train>0,dim=1)
                forward = labelrl_train>0
                forward[torch.arange(len(action),device=device),y] = 0
                forward = torch.cat([forward[:,1:],x.unsqueeze(1)],dim=1)
                outp = out_new.clone().detach().softmax(dim=-1)[:,0].log() 
                #print(len(out_new),forward.sum(),(seqrl[:,len_]>0).sum())
                R_item[forward] = outp
                R_item = R_item - R_itemb * (forward)
                th = min(0.8 + epoch * 0.03,0.99)
                R_item[:,:-1][p[:,1:]< th] = 0
                R_item[:,-1]= 0


                beta = torch.arange(action.shape[1],device=device).unsqueeze(0)
                gamma = torch.pow(args.gamma,beta)
                R_item = R_item * gamma
                R_item= torch.cumsum(R_item.flip(-1),dim=1).flip(-1) / gamma
                R_item[seq==0] = 0

                ''' Reward PP '''



                R_target = torch.zeros(action.shape,device=device)
                R_target[label>0] = logprob
                R_target = torch.cat([torch.zeros(len(R_target),1,device = device) ,R_target[:,:-1]],dim=1) 
                R_target_b = (R_target.sum(dim=-1) / (R_target!=0).float().sum(dim=-1)).unsqueeze(-1) -  baserate
                R_target = R_target - R_target_b
                R_target = (2 * action.float() - 1) * R_target
                R_target[seq==0] = 0
                R = R_item * args.ld1  + R_target * args.ld2 #* (epoch+1)

            pr = torch.where(action,p,1-p)
            pr = -(pr+1e-12).log()
            lossrl = (R * pr).mean()
            
            opt.zero_grad()
            opt_sampler.zero_grad()
            (lossrl+loss).backward()
            opt.step()
            opt_sampler.step()
            
            with torch.no_grad():
                total += loss.cpu().item()
                pbar.set_postfix({'L':total/(idx+1)})
                    
          


            if step % args.eval_per_steps == 0:
                model.eval()
                sampler.eval()
                with torch.no_grad():
           
                    m = eval(eval_loader,model,sampler,[20,10],args.num_item,args.device,tau) 
             
                    print('validation',epoch,m)
                    f = open(args.save_path + 'result.txt','a+')
                    print(epoch,m,file = f)
                    if m['NDCG@10'] > best:
                        g = open(args.save_path +'model'+str(args.t)+str(baserate)+'.pkl','wb')
                        pickle.dump(model,g)
                        g.close()
                        g = open(args.save_path +'sampler'+str(args.t)+str(baserate)+'.pkl','wb')
                        pickle.dump(sampler,g)
                        g.close()
                        best = m['NDCG@10']
                        stop = 0
                    else:
                        stop += 1
                        if stop == 1: return
                    f.close()
                model.train()
                sampler.train()
        f = open(args.save_path + 'loss.txt','a+')
        print('epoch',epoch,'loss',total/(idx+1),file=f)
        f.close()

if __name__ == '__main__':
    from args import args
    main(args)
