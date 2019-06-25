# Понимание сетей LSTM
## Recurrent Neural Networks

Люди не начинают думать с нуля каждую секунду. Когда вы читаете это эссе, вы понимаете каждое слово на основе вашего понимания предыдущих слов. Вы не выбрасываете все и снова начинаете думать с нуля. Ваши мысли настойчивы.

Традиционные нейронные сети не могут этого сделать, и это кажется серьезным недостатком. Например, представьте, что вы хотите классифицировать, какое событие происходит в каждой точке фильма. Неясно, как традиционная нейронная сеть могла бы использовать свои рассуждения о предыдущих событиях в фильме для информирования более поздних.

Периодические нейронные сети решают эту проблему. Это сети с петлями в них, позволяющими сохранять информацию.

 ![](/images/684601aa63886d86a1b4dafcf8ab079c.png)  
**Периодические нейронные сети имеют петли.**

На приведенной выше диаграмме блок нейронной сети \\(A\\) просматривает некоторый вход \\(x\ _t\\) и выводит значение \\(h\_t\\). Цикл позволяет передавать информацию от одного шага сети к следующему.

Эти петли заставляют повторяющиеся нейронные сети казаться загадочными. Однако, если вы думаете немного больше, оказывается, что они ничем не отличаются от обычной нейронной сети. Периодическая нейронная сеть может рассматриваться как несколько копий одной и той же сети, каждая из которых передает сообщение преемнику. Посмотрим, что произойдет, если мы развернем цикл:

![An unrolled recurrent neural network.](/images/34a870b0e60d513e7153b3f27fa66786.png)  
**развернутая рекуррентная нейронная сеть.**

Эта цепочечная природа показывает, что рекуррентные нейронные сети тесно связаны с последовательностями и списками. Это естественная архитектура нейронной сети, используемая для таких данных.

И они, безусловно, используются! За последние несколько лет был достигнут невероятный успех в применении RNN для решения разнообразных задач: распознавания речи, языкового моделирования, перевода, субтитров изображения... Список можно продолжать. Я оставлю обсуждение удивительных подвигов, которых можно достичь с помощью RNN, замечательному посту Андрея Карпати, [Необоснованная эффективность рекуррентных нейронных сетей](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). Но они действительно потрясающие.

Существенным для этих успехов является использование «LSTM», очень особого вида рекуррентной нейронной сети, которая для многих задач работает намного лучше, чем стандартная версия. Почти все захватывающие результаты, основанные на повторяющихся нейронных сетях, достигаются с их помощью. Именно эти LSTMs, которые исследует это эссе.

## Проблема долгосрочных зависимостей

Одна из привлекательных сторон сетей RNN заключается в том, что они могут подключить предыдущую информацию к текущей задаче, например, использование предыдущих видеокадров может дать понимание понимания текущего кадра. Если бы RNN могли сделать это, они были бы чрезвычайно полезны. Но могут ли они? Это зависит.

Иногда нам нужно только взглянуть на недавнюю информацию, чтобы выполнить текущую задачу. Например, рассмотрим языковую модель, пытающуюся предсказать следующее слово на основе предыдущих. Если мы пытаемся предсказать последнее слово в «облаках в _sky_», нам не нужен какой-либо дальнейший контекст - вполне очевидно, что следующим словом будет небо. В таких случаях, когда разрыв между соответствующей информацией и местом, в котором она необходима, невелик, RNN могут научиться использовать прошлую информацию.

 ![](/images/724834cc384c9fae8c3b3aff0d4a7a3a.png) 

Но есть также случаи, когда нам нужно больше контекста. Подумайте о том, чтобы попытаться предсказать последнее слово в тексте: «Я вырос во Франции. Я свободно говорю по-французски.» Недавняя информация говорит о том, что следующим словом, вероятно, является название языка, но если мы хотим сузить язык, нам нужен контекст Франции, если смотреть дальше. Вполне возможно, что разрыв между соответствующей информацией и точкой, в которой она необходима, станет очень большим.

К сожалению, по мере того, как этот разрыв увеличивается, RNN становятся неспособными научиться соединять информацию.

 ![Neural networks struggle with long term dependencies.](/images/6b911b2e1cb16cf47bf0f86257912c8e.png) 

Теоретически, RNN абсолютно способны обрабатывать такие «долгосрочные зависимости». Человек может тщательно подобрать параметры для них, чтобы решить игрушечные проблемы этой формы. К сожалению, на практике RNN, кажется, не в состоянии изучить их. Проблема была глубоко изучена [Hochreiter (1991) \[German\]](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) а также [Bengio, et al. (1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf), кто нашел несколько довольно фундаментальных причин, почему это может быть трудно.

К счастью, у LSTM нет этой проблемы!

## LSTM Сети

Сети с долговременной кратковременной памятью, обычно называемые «LSTM», представляют собой особый тип RNN, способный изучать долгосрочные зависимости. Они были представлены [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf) и были усовершенствованы и популяризированы многими людьми в следующей работе. [1](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#fn1) Они отлично работают по широкому кругу проблем и в настоящее время широко используются.

LSTM явно разработаны, чтобы избежать проблемы долгосрочной зависимости. Запоминание информации в течение длительных периодов времени - это практически поведение по умолчанию, а не то, чему они пытаются научиться!

Все рекуррентные нейронные сети имеют форму цепочки повторяющихся модулей нейронной сети. В стандартных RNN этот повторяющийся модуль будет иметь очень простую структуру, такую как один слой tanh.

 ![](/images/bce0d99e8bc969fa3c2ffa99f93935c5.png)  **The repeating module in a standard RNN contains a single layer.** 

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

 ![A LSTM neural network.](/images/2795bc16b012322f7767cd4d940ba2e3.png)  **The repeating module in an LSTM contains four interacting layers.** 

Don’t worry about the details of what’s going on. We’ll walk through the LSTM diagram step by step later. For now, let’s just try to get comfortable with the notation we’ll be using.

 ![](/images/1aed12786463eade9585fd20d894f49c.png) 

In the above diagram, each line carries an entire vector, from the output of one node to the inputs of others. The pink circles represent pointwise operations, like vector addition, while the yellow boxes are learned neural network layers. Lines merging denote concatenation, while a line forking denote its content being copied and the copies going to different locations.

## The Core Idea Behind LSTMs

The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

 ![](/images/1c663cbd3de24a90a0a94011fda6defd.png) 

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.

Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

 ![](/images/6ba1445193a5731e297922efdde6559f.png) 

The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

An LSTM has three of these gates, to protect and control the cell state.

## Step-by-Step LSTM Walk Through

The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at \\(h\_{t-1}\\) and \\(x\_t\\) , and outputs a number between \\(0\\) and \\(1\\) for each number in the cell state \\(C\_{t-1}\\) . A \\(1\\) represents “completely keep this” while a \\(0\\) represents “completely get rid of this.”

Let’s go back to our example of a language model trying to predict the next word based on all the previous ones. In such a problem, the cell state might include the gender of the present subject, so that the correct pronouns can be used. When we see a new subject, we want to forget the gender of the old subject.

 ![](/images/4035f737ddf1b26add27f4f69ccc1483.png) 

The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, \\(\\tilde{C}\_t\\) , that could be added to the state. In the next step, we’ll combine these two to create an update to the state.

In the example of our language model, we’d want to add the gender of the new subject to the cell state, to replace the old one we’re forgetting.

 ![](/images/5a460ea6f112de7332ca9584300c6e9c.png) 

It’s now time to update the old cell state, \\(C\_{t-1}\\) , into the new cell state \\(C\_t\\) . The previous steps already decided what to do, we just need to actually do it.

We multiply the old state by \\(f\_t\\) , forgetting the things we decided to forget earlier. Then we add \\(i\_t\*\\tilde{C}\_t\\) . This is the new candidate values, scaled by how much we decided to update each state value.

In the case of the language model, this is where we’d actually drop the information about the old subject’s gender and add the new information, as we decided in the previous steps.

 ![](/images/830f08d8fff0c45ff6350c8c473b50ba.png) 

Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through \\(\\tanh\\) (to push the values to be between \\(-1\\) and \\(1\\) ) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

For the language model example, since it just saw a subject, it might want to output information relevant to a verb, in case that’s what is coming next. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follows next.

 ![](/images/c3c06077f1a4e436ac442e0623ac284e.png) 

## Variants on Long Short Term Memory

What I’ve described so far is a pretty normal LSTM. But not all LSTMs are the same as the above. In fact, it seems like almost every paper involving LSTMs uses a slightly different version. The differences are minor, but it’s worth mentioning some of them.

One popular LSTM variant, introduced by [Gers & Schmidhuber (2000)](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf) , is adding “peephole connections.” This means that we let the gate layers look at the cell state.

 ![](/images/838ddfa82b7c6886c3687fa69574c212.png) 

The above diagram adds peepholes to all the gates, but many papers will give some peepholes and not others.

Another variation is to use coupled forget and input gates. Instead of separately deciding what to forget and what we should add new information to, we make those decisions together. We only forget when we’re going to input something in its place. We only input new values to the state when we forget something older.

 ![](/images/89bf31964d7553fa78bba9711c3d24e3.png) 

A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by [Cho, et al. (2014)](http://arxiv.org/pdf/1406.1078v3.pdf) . It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models, and has been growing increasingly popular.

 ![A gated recurrent unit neural network.](/images/f2716bc289734d8b545926b38a224692.png) 

These are only a few of the most notable LSTM variants. There are lots of others, like Depth Gated RNNs by [Yao, et al. (2015)](http://arxiv.org/pdf/1508.03790v2.pdf) . There’s also some completely different approach to tackling long-term dependencies, like Clockwork RNNs by [Koutnik, et al. (2014)](http://arxiv.org/pdf/1402.3511v1.pdf) .

Which of these variants is best? Do the differences matter? [Greff, et al. (2015)](http://arxiv.org/pdf/1503.04069.pdf) do a nice comparison of popular variants, finding that they’re all about the same. [Jozefowicz, et al. (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) tested more than ten thousand RNN architectures, finding some that worked better than LSTMs on certain tasks.

## Conclusion

Earlier, I mentioned the remarkable results people are achieving with RNNs. Essentially all of these are achieved using LSTMs. They really work a lot better for most tasks!

Written down as a set of equations, LSTMs look pretty intimidating. Hopefully, walking through them step by step in this essay has made them a bit more approachable.

LSTMs were a big step in what we can accomplish with RNNs. It’s natural to wonder: is there another big step? A common opinion among researchers is: “Yes! There is a next step and it’s attention!” The idea is to let every step of an RNN pick information to look at from some larger collection of information. For example, if you are using an RNN to create a caption describing an image, it might pick a part of the image to look at for every word it outputs. In fact, [Xu, _et al._ (2015)](http://arxiv.org/pdf/1502.03044v2.pdf) do exactly this – it might be a fun starting point if you want to explore attention! There’s been a number of really exciting results using attention, and it seems like a lot more are around the corner…

Attention isn’t the only exciting thread in RNN research. For example, Grid LSTMs by [Kalchbrenner, _et al._ (2015)](http://arxiv.org/pdf/1507.01526v1.pdf) seem extremely promising. Work using RNNs in generative models – such as [Gregor, _et al._ (2015)](http://arxiv.org/pdf/1502.04623.pdf) , [Chung, _et al._ (2015)](http://arxiv.org/pdf/1506.02216v3.pdf) , or [Bayer & Osendorfer (2015)](http://arxiv.org/pdf/1411.7610v3.pdf) – also seems very interesting. The last few years have been an exciting time for recurrent neural networks, and the coming ones promise to only be more so!

## Acknowledgments

I’m grateful to a number of people for helping me better understand LSTMs, commenting on the visualizations, and providing feedback on this post.

I’m very grateful to my colleagues at Google for their helpful feedback, especially [Oriol Vinyals](http://research.google.com/pubs/OriolVinyals.html) , [Greg Corrado](http://research.google.com/pubs/GregCorrado.html) , [Jon Shlens](http://research.google.com/pubs/JonathonShlens.html) , [Luke Vilnis](http://people.cs.umass.edu/~luke/) , and [Ilya Sutskever](http://www.cs.toronto.edu/~ilya/) . I’m also thankful to many other friends and colleagues for taking the time to help me, including [Dario Amodei](https://www.linkedin.com/pub/dario-amodei/4/493/393) , and [Jacob Steinhardt](http://cs.stanford.edu/~jsteinhardt/) . I’m especially thankful to [Kyunghyun Cho](http://www.kyunghyuncho.me/) for extremely thoughtful correspondence about my diagrams.

Before this post, I practiced explaining LSTMs during two seminar series I taught on neural networks. Thanks to everyone who participated in those for their patience with me, and for their feedback.

* * *

1.  In addition to the original authors, a lot of people contributed to the modern LSTM. A non-comprehensive list is: Felix Gers, Fred Cummins, Santiago Fernandez, Justin Bayer, Daan Wierstra, Julian Togelius, Faustino Gomez, Matteo Gagliolo, and [Alex Graves](https://scholar.google.com/citations?user=DaFHynwAAAAJ&hl=en) . [↩](https://colah.github.io/posts/2015-08-Understanding-LSTMs/#fnref1)






**********
[LSTM](/tags/LSTM.md)
