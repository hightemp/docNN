# Нейронный машинный перевод: использование Open-NMT для обучения модели перевода

Этот блог предназначен для предоставления пошагового руководства, чтобы научиться генерировать переводы с данного языка на любой целевой язык. Методология, которую мы используем для поставленной задачи, полностью мотивирована библиотекой с открытым исходным кодом, реализация pyTorch которой доступна на языке python, называемой Open-NMT (Open-Source Neural Machine Translation). Он разработан для того, чтобы энтузиасты глубокого обучения могли его использовать для реализации своих идей в области машинного перевода, обобщения, преобразования изображений в текст, морфологии и т.д.

Хотя существует довольно много эффективных систем перевода от Google Translate, Microsoft и т. д., Они либо не имеют открытого исходного кода, либо закрыты по ограничительной лицензии. Другие библиотеки, такие как модели tennflow-seq2seq, существуют для этой цели, но в качестве исследовательского кода. Open-NMT не только с открытым исходным кодом, но также предоставляет хорошо документированный, модульный и читаемый код для быстрого обучения и эффективной работы моделей.

 ![](/images/24bcf281705c0ffc16d6e2d37c12e317.jpeg) 

We elaborate further, a detailed guide for setting up the library and using the toolkit for training your own custom translation system. This blog deals with generating translation for Hindi language from given English text.

### A Brief Overview of the Architecture of Open-NMT :

Open-NMT is based on the research by Guillaume Klein et al, found [here](http://aclweb.org/anthology/P17-4012) .

 ![](/images/b0b432cc8cbc84c30d186e9eca18e8c9.png) 

According to the Paper, the following details are revealed about its architecture :

> OpenNMT is a complete library for training and deploying neural machine translation models. The system is successor to seq2seq-attn developed at Harvard, and has been completely rewritten for ease of efficiency, readability, and generalizability. It includes vanilla NMT models along with support for attention, gating, stacking, input feeding, regularization, beam search and all other options necessary for state-of-the-art performance.

> The main system is implemented in the Lua/Torch mathematical framework, and can be easily be extended using Torch’s internal standard neural network components. It has also been extended by Adam Lerer of Facebook Research to support Python/PyTorch framework, with the same API.

### Setup of Required Modules

The chief package required for training your custom translation system is essentially pyTorch, in which the Open-NMT models have been implemented.

The priliminary step, of course is to clone the OpenNMT-py repository :

git clone https://github.com/OpenNMT/OpenNMT-py  
cd OpenNMT-py

Here’s a requirements.txt file to gather all the required packages :

six  
tqdm  
torch>=0.4.0  
git+ [https://github.com/pytorch/text](https://github.com/pytorch/text)   
future

Since PyTorch is has been continuously evolving, we recommend forking the PyTorch 0.4 version to ensure a stable performance of the code base.

Run the following command to automatically gather pre-requisite dependencies :

pip install -r requirements.txt

### Gather the Datasets

The dataset comprises of a parallel corpus of source and target language files containing one sentence per line such that each tokens are separated by a space.

For our tutorial, we use a parallel corpora of English and Hindi sentences stored in separate files. The data is gathered from various sources and combined. The data is then re-arranged so as to create a set of files as follows :

*    `src-train.txt : Training file containing 10000 English (Source Language) sentences` 
*    `tgt-train.txt : Training file containing 10000 Hindi (Target Language) sentences` 
*    `src-val.txt : Validation data consisting of 1000 English (Source Language) sentences` 
*    `tgt-val.txt : Validation data consisting of 1000 Hindi (Target Language) sentences` 
*    `src-test.txt : Test Evaluation data consisting of 1000 English (Source Language) sentences` 
*    `tgt-test.txt : Test Evaluation data consisting of 1000 Hindi (Target Language) sentences` 

All of the above files are placed in the /data directory.

 **NOTE :** We have used a limited amount of data for explanation and experimentation in this tutorial. However it is recommended to use a large corpora with millions of sentences to ensure a vast vocabulary of unique words for better learning and a close-to-human-translation.

Validation data is employed to evaluate the model at each step to identify the convergence point. It should contain a maximum of 5000 sentences typically.

Here’s a sample showing how the text data is arranged in the corresponding files :

Source Files :  
They also bring out a number of Tamil weekly newspapers.  
They are a hard — working people and most of them work as labourers.  
Tamil films are also shown in the local cinema halls.  
There are quite a large number of Malayalees living here.

Target Files :

तमिल भाषा में वे अनेक समाचार पत्र व पत्रिकाएं भी निकालते हैं .  
ये लोग काफी परिश्रमी हैं , अधिकांश लोग मजदूरी करते हैं .  
स्थानीय सिनेमा हालों में तमिल चलचित्रों का प्रदर्शन अक्सर किया जाता है .  
मलयालम लोगों की बहुत बडी संख्या है .

### Pre-Processing Text Data :

Execute the following command for pre-processing the training and validation data and extract features for training and generate vocabulary files for the model.

python preprocess.py -train\_src data/src-train.txt -train\_tgt data/tgt-train.txt -valid\_src data/src-val.txt -valid\_tgt data/tgt-val.txt -save\_data data/demo

### Training the Translator model :

The chief command for trianing is really simple to use. Essentially, it takes as input, a data file and a save file.

A summary of the default model used is as follows :

NMTModel(  
  (encoder): RNNEncoder(  
    (embeddings): Embeddings(  
      (make\_embedding): Sequential(  
        (emb\_luts): Elementwise(  
          (0): Embedding(20351, 500, padding\_idx=1)  
        )  
      )  
    )  
    (rnn): LSTM(500, 500, num\_layers=2, dropout=0.3)  
  )  
  (decoder): InputFeedRNNDecoder(  
    (embeddings): Embeddings(  
      (make\_embedding): Sequential(  
        (emb\_luts): Elementwise(  
          (0): Embedding(20570, 500, padding\_idx=1)  
        )  
      )  
    )  
    (dropout): Dropout(p=0.3)  
    (rnn): StackedLSTM(  
      (dropout): Dropout(p=0.3)  
      (layers): ModuleList(  
        (0): LSTMCell(1000, 500)  
        (1): LSTMCell(500, 500)  
      )  
    )  
    (attn): GlobalAttention(  
      (linear\_in): Linear(in\_features=500, out\_features=500, bias=False)  
      (linear\_out): Linear(in\_features=1000, out\_features=500, bias=False)  
      (softmax): Softmax()  
      (tanh): Tanh()  
    )  
  )  
  (generator): Sequential(  
    (0): Linear(in\_features=500, out\_features=20570, bias=True)  
    (1): LogSoftmax()  
  )  
)

* * *

python train.py -data data/demo -save\_model demo-model

The above command will run a default model, comprising a two layer LSTM possessing 500 hidden units for both, the encoder and the decoder. To specify utilization of your GPU for training, specify the -gpuid argument in the above command (say -gpuid 1 for specifying usage og GPU 1).

Typically, the default model goes on till 100000 epochs, such that a check-point is saved after every 5000 epochs. So if your model converges and the validation accuracy reaches a stable point earlier, you can stop further training and use the previously saved checkpoint.

### Translate your own Data :

 ![](/images/fa47ec50e5ec206672d02ad6964a7b52.jpeg) 

The following command may be executed to perform an inference step on unseen text in the Source language (English) and generate corresponding translations predicted :

python translate.py -model demo-model\_XYZ.pt -src data/src-test.txt -output pred.txt -replace\_unk -verbose

This will generate the translated output and store the predictions into a file named `pred.txt` .

The model was trained for 10000 epochs on a NVIDIA GEFORCE 2GB GPU. Training on a CPU will require a very high computational cost, hence it is recommended to use a high end GPU for training the model with a large amount of data at a faster rate.

### Sample Predictions of the Model :

Shown below, are couple of examples for the Hindi translations generated for corresponding English sentences after training the model.

Trees are capable of absorbing more of carbon dioxide, thus maintaining equilibrium in the air composition .

PREDICTED : पेडों में कार्बन डाईआक्साइड के बुरे लोग इस प्रकार पेड - पौधे का प्रयोग करने के लिए मौजूद हैं .  
  
He has hope that the gods will throw good things from the heavens , upon them .

PREDICTED :वे उमीद है कि वे घर से कुछ नहीं बची हैं .

The Buddhist temple , the Dalai Lama Palace and dispensary of Tibet are tourist attractions here .

PREDICTED :यहां का बौद्ध मंदिर दलाई लामा का आवास तथा तिब्बती औषधालय स्थानिय लोगो में मिलता है .

He lets the hair grow long.

PREDICTED : वह अपने बढा लेता है .

As seen above, the predictions are not good enough yet with less training data to be used for any real time translation. For performing translations close to real world, the model has to be trained on on a large vocabulary and about a million sentences, which will parallely involve a lot of computational cost in terms of hardware requirements and training time.

### Evaluate your Trained Model :

#### Bilingual Evaluation Understudy Score

The Bilingual Evaluation Understudy Score, or BLEU Score, refers to an evaluation metric for the purpose of evaluating Machine Translation Systems by comparing a generated sentence to a reference sentence.

A perfect match in this comparison results in a BLEU score of 1.0, whereas a complete mismatch results in a BLEU score of 0.0.

The BLEU Score is a universally adapted metric for evaluating translation models as it is independent of language, simple to interpret and has high correlation with manual evaluation.

The BLEU score was proposed in a research conducted by Kishore Papineni, et al. “ [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf) “.

The BLEU SCore is generated after counting n-grams in the candidate translation matching with the n-grams in the reference text. Word order is not taken into account in this comparison.

So how would we define a n-gram? Let’s say a 1-gram or uni-gram would mean each individual token and a bi-gram would represent each pair of word.

The Code for calculating BLEU Scores, given your predicted candidate file and a reference file given in the GitHub repository, the link to whic is provided at the end of the blog.

Here’s how we run the code to evaluate the model :

python calculatebleu.py "pred.txt" "tgt-test.txt"

Where pred.txt is our candidate predicted translation file and tgt-test.txt is the file containing the actual translations in the target language.

Since our data vocabulary generated with 10k sentences consists only of a few thousand words, the BLEU Score we get on our prediction is quite poor (0.025).

NOTE : Since our primary aim is focussed at elaborating on the usage of Open-NMT , we use only a small dataset, which is why the evaluation of our predicted translation results is a poor BLEU Score. A BLEU Score around 0.5 implies a decent translation. Increase the training vocabulary manifold by adding several thousands of more examples to improve the score.

However Open-NMT allows us to train our own custom translator models between any pair of languages and is very convenient to use.

The Code for generating the BLEU Score and the datasets used in training our model have been provided [here](https://github.com/DataTurks-Engg/Neural_Machine_Translation) .

If you have any queries or suggestions, I would love to hear about it. Please write to me at abhishek.narayanan@dataturks.com.

**********
[OpenNMT](/tags/OpenNMT.md)
