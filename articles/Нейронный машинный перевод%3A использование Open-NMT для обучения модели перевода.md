# Нейронный машинный перевод: использование Open-NMT для обучения модели перевода

Этот блог предназначен для предоставления пошагового руководства, чтобы научиться генерировать переводы с данного языка на любой целевой язык. Методология, которую мы используем для поставленной задачи, полностью мотивирована библиотекой с открытым исходным кодом, реализация pyTorch которой доступна на языке python, называемой Open-NMT (Open-Source Neural Machine Translation). Он разработан для того, чтобы энтузиасты глубокого обучения могли его использовать для реализации своих идей в области машинного перевода, обобщения, преобразования изображений в текст, морфологии и т.д.

Хотя существует довольно много эффективных систем перевода от Google Translate, Microsoft и т. д., Они либо не имеют открытого исходного кода, либо закрыты по ограничительной лицензии. Другие библиотеки, такие как модели tennflow-seq2seq, существуют для этой цели, но в качестве исследовательского кода. Open-NMT не только с открытым исходным кодом, но также предоставляет хорошо документированный, модульный и читаемый код для быстрого обучения и эффективной работы моделей.

 ![](/images/24bcf281705c0ffc16d6e2d37c12e317.jpeg) 

Далее мы разрабатываем подробное руководство по настройке библиотеки и использованию инструментария для обучения вашей собственной системе перевода. Этот блог посвящен созданию перевода на хинди из данного английского текста.

### Краткий обзор архитектуры Open-NMT:

Open-NMT основан на исследованиях Гийома Кляйна и др., Найденных [here](http://aclweb.org/anthology/P17-4012) .

 ![](/images/b0b432cc8cbc84c30d186e9eca18e8c9.png) 

Согласно документу, о его архитектуре раскрываются следующие подробности:

> OpenNMT - это полная библиотека для обучения и развертывания моделей нейронного машинного перевода. Система является наследницей seq2seq-attn, разработанной в Гарварде, и была полностью переписана для простоты эффективности, удобочитаемости и обобщения. Он включает в себя ванильные модели NMT, а также поддержку внимания, стробирования, укладки, подачи входного сигнала, регуляризации, поиска луча и всех других опций, необходимых для современного уровня производительности.

> Основная система реализована в математической структуре Lua / Torch и может быть легко расширена с помощью внутренних стандартных компонентов нейронной сети Torch. Он также был расширен Адамом Лерером из Facebook Research для поддержки платформы Python / PyTorch с тем же API.

### Настройка необходимых модулей

Главный пакет, необходимый для обучения вашей пользовательской системе перевода, - это, по сути, pyTorch, в котором были реализованы модели Open-NMT.

Конечно, предварительным шагом является клонирование репозитория OpenNMT-py:

```
git clone https://github.com/OpenNMT/OpenNMT-py  
cd OpenNMT-py
```

Вот файл require.txt для сбора всех необходимых пакетов:

```
six  
tqdm  
torch>=0.4.0  
git+ [https://github.com/pytorch/text](https://github.com/pytorch/text)   
future
```

Поскольку PyTorch постоянно развивается, мы рекомендуем использовать версию PyTorch 0.4, чтобы обеспечить стабильную производительность базы кода.

Выполните следующую команду, чтобы автоматически собрать обязательные зависимости:

```
pip install -r requirements.txt
```

### Соберите наборы данных

Набор данных состоит из параллельного корпуса файлов исходного и целевого языков, содержащих одно предложение в строке, так что каждый токен разделяется пробелом.

Для нашего урока мы используем параллельные корпусы предложений на английском и хинди, которые хранятся в отдельных файлах. Данные собираются из разных источников и объединяются. Затем данные переупорядочиваются так, чтобы создать набор файлов следующим образом:

* `src-train.txt : Учебный файл, содержащий 10000 предложений на английском языке (на исходном языке)` 
* `tgt-train.txt : Учебный файл, содержащий 10000 предложений на хинди (целевой язык)` 
* `src-val.txt : Данные проверки, состоящие из 1000 предложений на английском языке (на исходном языке)` 
* `tgt-val.txt : Данные проверки, состоящие из 1000 предложений на хинди (целевой язык)` 
* `src-test.txt : Данные тестовой оценки, состоящие из 1000 предложений на английском языке (на исходном языке)` 
* `tgt-test.txt : Данные оценки теста, состоящие из 1000 предложений на хинди (целевой язык)` 

Все вышеперечисленные файлы находятся в каталоге /data.

**Заметка:** В этом уроке мы использовали ограниченное количество данных для объяснения и экспериментов. Однако рекомендуется использовать большие корпуса с миллионами предложений, чтобы обеспечить обширный словарный запас уникальных слов для лучшего изучения и близкого к человеческому переводу.

Данные проверки используются для оценки модели на каждом этапе для определения точки сходимости. Обычно он должен содержать не более 5000 предложений.

Вот пример, показывающий, как текстовые данные расположены в соответствующих файлах:

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
