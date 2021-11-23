# HETFORMER

**This code is for paper `HETFORMER: Heterogeneous Transformer with Sparse Attention forLong-Text Extractive Summarization`**(https://arxiv.org/pdf/2110.06388.pdf)

**Python version**: This code is in Python3.6

**Package Requirements**: pytorch pytorch_pretrained_bert tensorboardX multiprocess pyrouge tensorboardX nlp rouge_score

Some codes are borrowed from BertSum(https://github.com/nlpyang/BertSum) and Longformer(https://github.com/allenai/longformer)

## download the processed data

download https://drive.google.com/drive/folders/1Bie3qagoyZe44uYTy8XkkGwfsi7HDeNb?usp=sharing

unzip the zipfile and put all `.pt` files into `cnndm_cluster` and `multinews_cluster`

## Model Training

**First run**: For the first time, you should use single-GPU, so the code can download the BERT model.
Change ``-visible_gpus 0,1,2 -gpu_ranks 0,1,2 -world_size 3`` to ``-visible_gpus 0 -gpu_ranks 0 -world_size 1``, after
downloading, you could kill the process and rerun the code with multi-GPUs.

To train the HETFORMER model, run:

```
python train.py
        -mode train 
        -encoder classifier 
        -dropout 0.1 
        -bert_data_path ../cnndm_cluster/cnndm 
        -model_path ../models/cnndm
        -lr 2e-3 -visible_gpus 0,1,2  
        -gpu_ranks 0,1,2 
        -world_size 1 
        -attention_window [16,16,32,32,64,64,128,128,256,256,256,256]
        -report_every 50 
        -save_checkpoint_steps 1000 
        -batch_size 3000 
        -decay_method noam 
        -train_steps 50000 
        -accum_count 2 
        -log_file ../logs/cnndm 
        -use_interval true 
        -warmup_steps 10000
```

* `-mode` can be {`train, validate, test`}, where `validate` will inspect the model directory and evaluate the model for
  each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* Change attention_windon to [64,64,128,128,256,256,512,512,512,512,512,512] on Multinews Dataset

## Model Test

After the training finished, run

```
python train.py 
        -mode test 
        -bert_data_path ../cnndm_cluster/cnndm 
        -test_from   MODEL_PATH
        -batch_size 30000   
        -result_path RESULT_PATH 
        -block_trigram true
        -top_n_sentence 3
```
* `MODEL_PATH` is the directory of saved checkpoints
* `RESULT_PATH` is where you want to put decoded summaries (default `../results/cnndm`)
* Change top_n_sentence to 9 in Multinews Dataset



