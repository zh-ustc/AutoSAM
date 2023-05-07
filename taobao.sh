for d in 0
do
                        python main.py \
                        --data_path ../../DataSample/taobao/ijcai2016seq100.csv \
                        --max_len 100 \
                        --save_path save/taobaot/ \
                        --lr 1e-3 \
                        --train_batch_size 128 \
                        --test_batch_size 64 \
                        --attn_heads 4 \
                        --dropout 0.1 \
                        --d_ffn 256 \
                        --d_model 128 \
                        --bert_layers 2 \
                        --ld1 $d \
                        --ld2 1e-3 \
                        --enable_res_parameter 1 \
                        --device cuda:3 \
                        --num_epoch 20 \
                        --baserate 2 \
                        --seed 2022
done

