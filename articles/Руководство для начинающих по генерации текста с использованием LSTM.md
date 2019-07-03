# Руководство для начинающих по генерации текста с использованием LSTM

Text Generation is a type of Language Modelling problem. Language Modelling is the core problem for a number of of natural language processing tasks such as speech to text, conversational system, and text summarization. A trained language model learns the likelihood of occurrence of a word based on the previous sequence of words used in the text. Language models can be operated at character level, n-gram level, sentence level or even paragraph level. In this notebook, I will explain how to create a language model for generating natural language text by implement and training state-of-the-art Recurrent Neural Network.

### Generating News headlines [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#Generating-News-headlines) 

In this kernel, I will be using the dataset of [New York Times Comments and Headlines](https://www.kaggle.com/aashita/nyt-comments) to train a text generation language model which can be used to generate News Headlines

## 1\. Import the libraries [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#1.-Import-the-libraries) 

As the first step, we need to import the required libraries:

In \[1\]:

 \# keras module for building LSTM  
 from   keras.preprocessing.sequence   import   pad\_sequences 
 from   keras.layers   import   Embedding  ,   LSTM  ,   Dense  ,   Dropout 
 from   keras.preprocessing.text   import   Tokenizer 
 from   keras.callbacks   import   EarlyStopping 
 from   keras.models   import   Sequential 
 import   keras.utils   as   ku  

 \# set seeds for reproducability 
 from   tensorflow   import   set\_random\_seed 
 from   numpy.random   import   seed 
 set\_random\_seed  (  2  ) 
 seed  (  1  ) 

 import   pandas   as   pd 
 import   numpy   as   np 
 import   string  ,   os  

 import   warnings 
 warnings  .  filterwarnings  (  "ignore"  ) 
 warnings  .  simplefilter  (  action  \=  'ignore'  ,   category  \=  FutureWarning  ) 

Using TensorFlow backend.

## 2\. Load the dataset [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#2.-Load-the-dataset) 

Load the dataset of news headlines

In \[2\]:

 curr\_dir   \=   '../input/' 
 all\_headlines   \=   \[\] 
 for   filename   in   os  .  listdir  (  curr\_dir  ): 
     if   'Articles'   in   filename  : 
         article\_df   \=   pd  .  read\_csv  (  curr\_dir   +   filename  ) 
         all\_headlines  .  extend  (  list  (  article\_df  .  headline  .  values  )) 
         break 

 all\_headlines   \=   \[  h   for   h   in   all\_headlines   if   h   !=   "Unknown"  \] 
 len  (  all\_headlines  ) 

Out\[2\]:

777

## 3\. Dataset preparation [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#3.-Dataset-preparation) 

### 3.1 Dataset cleaning [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#3.1-Dataset-cleaning) 

In dataset preparation step, we will first perform text cleaning of the data which includes removal of punctuations and lower casing all the words.

In \[3\]:

 def   clean\_text  (  txt  ): 
     txt   \=   ""  .  join  (  v   for   v   in   txt   if   v   not   in   string  .  punctuation  )  .  lower  () 
     txt   \=   txt  .  encode  (  "utf8"  )  .  decode  (  "ascii"  ,  'ignore'  ) 
     return   txt  

 corpus   \=   \[  clean\_text  (  x  )   for   x   in   all\_headlines  \] 
 corpus  \[:  10  \] 

Out\[3\]:

\[' gop leadership poised to topple obamas pillars',
 'fractured world tested the hope of a young president',
 'little troublemakers',
 'angela merkel russias next target',
 'boots for a stranger on a bus',
 'molder of navajo youth where a game is sacred',
 'the affair season 3 episode 6 noah goes home',
 'sprint and mr trumps fictional jobs',
 'america  becomes a stan',
 'fighting diabetes and leading by example'\]

### 3.2 Generating Sequence of N-gram Tokens [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#3.2-Generating-Sequence-of-N-gram-Tokens) 

Language modelling requires a sequence input data, as given a sequence (of words/tokens) the aim is the predict next word/token.

The next step is Tokenization. Tokenization is a process of extracting tokens (terms / words) from a corpus. Python’s library Keras has inbuilt model for tokenization which can be used to obtain the tokens and their index in the corpus. After this step, every text document in the dataset is converted into sequence of tokens.

In \[4\]:

 tokenizer   \=   Tokenizer  () 

 def   get\_sequence\_of\_tokens  (  corpus  ): 
     \## tokenization 
     tokenizer  .  fit\_on\_texts  (  corpus  ) 
     total\_words   \=   len  (  tokenizer  .  word\_index  )   +   1 
    
     \## convert data to sequence of tokens  
     input\_sequences   \=   \[\] 
     for   line   in   corpus  : 
         token\_list   \=   tokenizer  .  texts\_to\_sequences  (\[  line  \])\[  0  \] 
         for   i   in   range  (  1  ,   len  (  token\_list  )): 
             n\_gram\_sequence   \=   token\_list  \[:  i  +  1  \] 
             input\_sequences  .  append  (  n\_gram\_sequence  ) 
     return   input\_sequences  ,   total\_words 

 inp\_sequences  ,   total\_words   \=   get\_sequence\_of\_tokens  (  corpus  ) 
 inp\_sequences  \[:  10  \] 

Out\[4\]:

\[\[73, 313\],
 \[73, 313, 616\],
 \[73, 313, 616, 3\],
 \[73, 313, 616, 3, 617\],
 \[73, 313, 616, 3, 617, 205\],
 \[73, 313, 616, 3, 617, 205, 314\],
 \[618, 38\],
 \[618, 38, 619\],
 \[618, 38, 619, 1\],
 \[618, 38, 619, 1, 206\]\]

In the above output \[30, 507\], \[30, 507, 11\], \[30, 507, 11, 1\] and so on represents the ngram phrases generated from the input data. where every integer corresponds to the index of a particular word in the complete vocabulary of words present in the text. For example

 **Headline:** i stand with the shedevils  
 **Ngrams:** | **Sequence of Tokens** 

<table style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border-collapse: collapse; border-spacing: 0px; background: rgb(255, 255, 255); margin: 16px 0px 24px; border-width: 0px 1px 0px 0px; border-top-style: initial; border-right-style: solid; border-bottom-style: initial; border-left-style: initial; border-top-color: initial; border-right-color: rgba(222, 223, 224, 0.5); border-bottom-color: initial; border-left-color: initial; border-image: initial; color: rgba(0, 0, 0, 0.7); display: block; font-size: 12px; font-variant-numeric: tabular-nums; overflow: auto;"><tbody style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased;"><tr style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border: 1px solid black; border-collapse: collapse; margin: 1em 2em; background: rgb(255, 255, 255);"><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">Ngram</td><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">Sequence of Tokens</td></tr><tr style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border: 1px solid black; border-collapse: collapse; margin: 1em 2em; background: rgb(255, 255, 255);"><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">i stand</td><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">[30, 507]</td></tr><tr style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border: 1px solid black; border-collapse: collapse; margin: 1em 2em; background: rgb(255, 255, 255);"><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">i stand with</td><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">[30, 507, 11]</td></tr><tr style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border: 1px solid black; border-collapse: collapse; margin: 1em 2em; background: rgb(255, 255, 255);"><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">i stand with the</td><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">[30, 507, 11, 1]</td></tr><tr style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border: 1px solid black; border-collapse: collapse; margin: 1em 2em; background: rgb(255, 255, 255);"><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">i stand with the shedevils</td><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">[30, 507, 11, 1, 975]</td></tr></tbody></table>

### 3.3 Padding the Sequences and obtain Variables : Predictors and Target [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#3.3-Padding-the-Sequences-and-obtain-Variables-:-Predictors-and-Target) 

Now that we have generated a data-set which contains sequence of tokens, it is possible that different sequences have different lengths. Before starting training the model, we need to pad the sequences and make their lengths equal. We can use pad\_sequence function of Kears for this purpose. To input this data into a learning model, we need to create predictors and label. We will create N-grams sequence as predictors and the next word of the N-gram as label. For example:

Headline: they are learning data science

<table style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border-collapse: collapse; border-spacing: 0px; background: rgb(255, 255, 255); margin: 16px 0px 24px; border-width: 0px 1px 0px 0px; border-top-style: initial; border-right-style: solid; border-bottom-style: initial; border-left-style: initial; border-top-color: initial; border-right-color: rgba(222, 223, 224, 0.5); border-bottom-color: initial; border-left-color: initial; border-image: initial; color: rgba(0, 0, 0, 0.7); display: block; font-size: 12px; font-variant-numeric: tabular-nums; overflow: auto;"><tbody style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased;"><tr style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border: 1px solid black; border-collapse: collapse; margin: 1em 2em; background: rgb(255, 255, 255);"><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">PREDICTORS</td><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">LABEL</td></tr><tr style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border: 1px solid black; border-collapse: collapse; margin: 1em 2em; background: rgb(255, 255, 255);"><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">they</td><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">are</td></tr><tr style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border: 1px solid black; border-collapse: collapse; margin: 1em 2em; background: rgb(255, 255, 255);"><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">they are</td><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">learning</td></tr><tr style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border: 1px solid black; border-collapse: collapse; margin: 1em 2em; background: rgb(255, 255, 255);"><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">they are learning</td><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">data</td></tr><tr style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; border: 1px solid black; border-collapse: collapse; margin: 1em 2em; background: rgb(255, 255, 255);"><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">they are learning data</td><td style="box-sizing: border-box; text-rendering: auto; -webkit-font-smoothing: antialiased; padding: 4px 8px; border: 1px solid rgba(222, 223, 224, 0.5); border-collapse: collapse; margin: 1em 2em; text-align: left; vertical-align: middle;">science</td></tr></tbody></table>

In \[5\]:

 def   generate\_padded\_sequences  (  input\_sequences  ): 
     max\_sequence\_len   \=   max  (\[  len  (  x  )   for   x   in   input\_sequences  \]) 
     input\_sequences   \=   np  .  array  (  pad\_sequences  (  input\_sequences  ,   maxlen  \=  max\_sequence\_len  ,   padding  \=  'pre'  )) 
    
     predictors  ,   label   \=   input\_sequences  \[:,:  \-  1  \],  input\_sequences  \[:,  \-  1  \] 
     label   \=   ku  .  to\_categorical  (  label  ,   num\_classes  \=  total\_words  ) 
     return   predictors  ,   label  ,   max\_sequence\_len 

 predictors  ,   label  ,   max\_sequence\_len   \=   generate\_padded\_sequences  (  inp\_sequences  ) 

Perfect, now we can obtain the input vector X and the label vector Y which can be used for the training purposes. Recent experiments have shown that recurrent neural networks have shown a good performance in sequence to sequence learning and text data applications. Lets look at them in brief.

## 4\. LSTMs for Text Generation [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#4.-LSTMs-for-Text-Generation) 

 ![](/images/731ef175a2f91faac9d2045489ddfa35.png) 

Unlike Feed-forward neural networks in which activation outputs are propagated only in one direction, the activation outputs from neurons propagate in both directions (from inputs to outputs and from outputs to inputs) in Recurrent Neural Networks. This creates loops in the neural network architecture which acts as a ‘memory state’ of the neurons. This state allows the neurons an ability to remember what have been learned so far.

The memory state in RNNs gives an advantage over traditional neural networks but a problem called Vanishing Gradient is associated with them. In this problem, while learning with a large number of layers, it becomes really hard for the network to learn and tune the parameters of the earlier layers. To address this problem, A new type of RNNs called LSTMs (Long Short Term Memory) Models have been developed.

LSTMs have an additional state called ‘cell state’ through which the network makes adjustments in the information flow. The advantage of this state is that the model can remember or forget the leanings more selectively. To learn more about LSTMs, here is a great post. Lets architecture a LSTM model in our code. I have added total three layers in the model.

1.  Input Layer : Takes the sequence of words as input
2.  LSTM Layer : Computes the output using LSTM units. I have added 100 units in the layer, but this number can be fine tuned later.
3.  Dropout Layer : A regularisation layer which randomly turns-off the activations of some neurons in the LSTM layer. It helps in preventing over fitting. (Optional Layer)
4.  Output Layer : Computes the probability of the best possible next word as output

We will run this model for total 100 epoochs but it can be experimented further.

In \[6\]:

 def   create\_model  (  max\_sequence\_len  ,   total\_words  ): 
     input\_len   \=   max\_sequence\_len   \-   1 
     model   \=   Sequential  () 
    
     \# Add Input Embedding Layer 
     model  .  add  (  Embedding  (  total\_words  ,   10  ,   input\_length  \=  input\_len  )) 
    
     \# Add Hidden Layer 1 - LSTM Layer 
     model  .  add  (  LSTM  (  100  )) 
     model  .  add  (  Dropout  (  0.1  )) 
    
     \# Add Output Layer 
     model  .  add  (  Dense  (  total\_words  ,   activation  \=  'softmax'  )) 

     model  .  compile  (  loss  \=  'categorical\_crossentropy'  ,   optimizer  \=  'adam'  ) 
    
     return   model 

 model   \=   create\_model  (  max\_sequence\_len  ,   total\_words  ) 
 model  .  summary  () 

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
Layer (type)                 Output Shape              Param #   
=================================================================
embedding\_1 (Embedding)      (None, 20, 10)            22170     
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
lstm\_1 (LSTM)                (None, 100)               44400     
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dropout\_1 (Dropout)          (None, 100)               0         
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
dense\_1 (Dense)              (None, 2217)              223917    
=================================================================
Total params: 290,487
Trainable params: 290,487
Non-trainable params: 0
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Lets train our model now

In \[7\]:

 model  .  fit  (  predictors  ,   label  ,   epochs  \=  100  ,   verbose  \=  5  ) 

Epoch 1/100
Epoch 2/100
Epoch 3/100
Epoch 4/100
Epoch 5/100
Epoch 6/100
Epoch 7/100
Epoch 8/100
Epoch 9/100
Epoch 10/100
Epoch 11/100
Epoch 12/100
Epoch 13/100
Epoch 14/100
Epoch 15/100
Epoch 16/100
Epoch 17/100
Epoch 18/100
Epoch 19/100
Epoch 20/100
Epoch 21/100
Epoch 22/100
Epoch 23/100
Epoch 24/100
Epoch 25/100
Epoch 26/100
Epoch 27/100
Epoch 28/100
Epoch 29/100
Epoch 30/100
Epoch 31/100
Epoch 32/100
Epoch 33/100
Epoch 34/100
Epoch 35/100
Epoch 36/100
Epoch 37/100
Epoch 38/100
Epoch 39/100
Epoch 40/100
Epoch 41/100
Epoch 42/100
Epoch 43/100
Epoch 44/100
Epoch 45/100
Epoch 46/100
Epoch 47/100
Epoch 48/100
Epoch 49/100
Epoch 50/100
Epoch 51/100
Epoch 52/100
Epoch 53/100
Epoch 54/100
Epoch 55/100
Epoch 56/100
Epoch 57/100
Epoch 58/100
Epoch 59/100
Epoch 60/100
Epoch 61/100
Epoch 62/100
Epoch 63/100
Epoch 64/100
Epoch 65/100
Epoch 66/100
Epoch 67/100
Epoch 68/100
Epoch 69/100
Epoch 70/100
Epoch 71/100
Epoch 72/100
Epoch 73/100
Epoch 74/100
Epoch 75/100
Epoch 76/100
Epoch 77/100
Epoch 78/100
Epoch 79/100
Epoch 80/100
Epoch 81/100
Epoch 82/100
Epoch 83/100
Epoch 84/100
Epoch 85/100
Epoch 86/100
Epoch 87/100
Epoch 88/100
Epoch 89/100
Epoch 90/100
Epoch 91/100
Epoch 92/100
Epoch 93/100
Epoch 94/100
Epoch 95/100
Epoch 96/100
Epoch 97/100
Epoch 98/100
Epoch 99/100
Epoch 100/100

Out\[7\]:

\<keras.callbacks.History at 0x7f2ddf540550>

## 5\. Generating the text [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#5.-Generating-the-text) 

Great, our model architecture is now ready and we can train it using our data. Next lets write the function to predict the next word based on the input words (or seed text). We will first tokenize the seed text, pad the sequences and pass into the trained model to get predicted word. The multiple predicted words can be appended together to get predicted sequence.

In \[8\]:

 def   generate\_text  (  seed\_text  ,   next\_words  ,   model  ,   max\_sequence\_len  ): 
     for   \_   in   range  (  next\_words  ): 
         token\_list   \=   tokenizer  .  texts\_to\_sequences  (\[  seed\_text  \])\[  0  \] 
         token\_list   \=   pad\_sequences  (\[  token\_list  \],   maxlen  \=  max\_sequence\_len  \-  1  ,   padding  \=  'pre'  ) 
         predicted   \=   model  .  predict\_classes  (  token\_list  ,   verbose  \=  0  ) 
        
         output\_word   \=   "" 
         for   word  ,  index   in   tokenizer  .  word\_index  .  items  (): 
             if   index   \==   predicted  : 
                 output\_word   \=   word 
                 break 
         seed\_text   +=   " "  +  output\_word 
     return   seed\_text  .  title  () 

## 6\. Some Results [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#6.-Some-Results) 

Code

United States On Paralysis Its A Workout
Preident Trump Fires We Be Mindful
Donald Trump Tweets Blacks Perceive A
India And China 3 Episode 7 Theres
New York Today A Trumpless Tower
Science And Technology Nam A Raid And A

## Improvement Ideas [](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms#Improvement-Ideas) 

As we can see, the model has produced the output which looks fairly fine. The results can be improved further with following points:

*   Adding more data
*   Fine Tuning the network architecture
*   Fine Tuning the network parameters

Thanks for going through the notebook, please upvote if you liked.

**********
[LSTM](/tags/LSTM.md)
