# AutoSAM

## Project Overview


This is an anonymously open-sourced project designed for scientific research and technological development, supporting the blind review process to ensure fairness and objectivity in evaluations. 

## Dataset

We provide a smaller dataset Yelp for testing. The dataset has been preprocessed into a user interaction sequence file named `seq_yelp.csv`, containing 221,397 users. Each line represents a time-ordered sequence of item IDs that the user has interacted with. 

The file can be extracted using the following command:

~~~python
unzip seq_yelp.csv.zip
~~~

## Instructions

```bash
# Run to train (SASRec + AutoSAM)
python main.py \
--data_path seq_yelp.csv \
--max_len 100 \
--save_path save/ \
--lr 1e-3 \
--train_batch_size 128 \
--test_batch_size 256 \
--attn_heads 4 \
--dropout 0.1 \
--d_ffn 256 \
--d_model 128 \
--bert_layers 2 \
--ld1 1e-2 \
--ld2 1e-2 \
--enable_res_parameter 1 \
--device cuda:0 \
--num_epoch 20 \
--baserate 1 \
--t 5 \
--seed 0

# Run to train baseline (SASRec)
python main_baseline.py \
--data_path seq_yelp.csv \
--max_len 100 \
--save_path save/ \
--lr 1e-3 \
--train_batch_size 128 \
--test_batch_size 256 \
--attn_heads 4 \
--dropout 0.1 \
--d_ffn 256 \
--d_model 128 \
--bert_layers 2 \
--enable_res_parameter 1 \
--device cuda:0 \
--num_epoch 20 \
--seed 0
```

## Experiment

We use the leave-one-out method to partition the dataset, training for approximately **7** **epochs** until convergence. Both our proposed method and the baseline consume about **5GB** of GPU memory on an A100, with a training time of approximately **100 seconds** per epoch. The results for AutoSAM and the baseline are as follows:

**Validation Set**

|                  | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
| ---------------- | ------- | ------- | --------- | --------- |
| SASRec           | 0.0277  | 0.0355  | 0.0539    | 0.0847    |
| SASRec + AutoSAM | 0.0296  | 0.0381  | 0.0572    | 0.0909    |

**Testing Set**

|                  | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
| ---------------- | ------- | ------- | --------- | --------- |
| SASRec           | 0.0245  | 0.0314  | 0.0473    | 0.0746    |
| SASRec + AutoSAM | 0.0272  | 0.0347  | 0.0521    | 0.0817    |
