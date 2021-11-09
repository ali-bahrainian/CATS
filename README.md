# CATS
### CATS: Customizable Abstractive Topic-based Summarization

This repository contains code related to the paper “CATS: Customizable Abstractive Topic-based Summarization” published at Transactions of Information Systems (TOIS) journal, 2021. 


The code has been developed using Python 3, and v1 compatible TensorFlow 2 code. This implementation is based on code releases related to Pointer-Generator Networks [here](https://github.com/abisee/pointer-generator) and the TextSum project. 


#### Dataset
Obtaining the non-anonymized CNN/DailyMail dataset Used in the Paper:
In order to obtain the dataset, we encourage users to download and preprocess the dataset as described [here](https://github.com/abisee/cnn-dailymail). Furthermore, we use the exact same setting of [chunked data](https://github.com/abisee/cnn-dailymail/issues/3). 
 

#### Using Topic Information:

The LDA models used in our paper can be obtained from [here](https://drive.google.com/drive/folders/1M86uAM21Zx8Xn-W4TTi74t7nWCQr1f9v?usp=sharing). The current code release has been tested with the 150 topics pre-trained LDA model. You can make a reference to one of the provided LDA topic models in data.py in the TopicModel class. By default, the LDA model is expected to be in a folder called `lda`, which contains `lda.model` and `dictionary.dic` available from the beforementioned URL.

#### Train 
In order to train the model you may run:

```
python3 run_summarization.py --mode=train --data_path='data/chunked/train_*' --vocab_path='data/vocab' --log_root='logs' --exp_name=myexperiment
```

This will create a subdirectory of your specified log_root called myexperiment where all checkpoints will be saved. Then the model will start training using the train_*.bin files as training data.


#### Decoding
As stated in the paper, no topic information were used at test time. In order to decode without topic information, we used the pointer-generator basic model code [here](https://github.com/abisee/pointer-generator). After downloading the code, you may decode using:

```
python3 run_summarization.py --mode=decode --data_path='data/chunked/val_*' --vocab_path='data/vocab' --log_root='logs' --exp_name=myexperiment
```

Please note that one should run the above command using the same settings entered for the training job (plus any decode mode specific flags like beam_size).

This will repeatedly load random examples from your specified datafile and generate a summary using beam search. The results will be printed to screen.

If you would like to run evaluation on the entire validation or test set and obtain ROUGE scores, set the flag single_pass=1. This will go through the entire dataset in the same order, writing the generated summaries to file, and then running evaluation using pyrouge.   
