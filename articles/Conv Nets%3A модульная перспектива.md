# Conv Nets: модульная перспектива
## Вступление

В последние несколько лет глубокие нейронные сети привели к прорывным результатам по ряду проблем распознавания образов, таких как компьютерное зрение и распознавание голоса. Одним из важных компонентов, приводящих к этим результатам, была особая разновидность нейронной сети, называемая _конволюционная нейронная сеть_.

По своей сути, сверточные нейронные сети можно рассматривать как разновидность нейронной сети, которая использует много идентичных копий одного и того же нейрона. [1](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/#fn1) Это позволяет сети иметь много нейронов и выражать вычислительно большие модели, сохраняя при этом количество фактических параметров - значения, описывающие, как ведут себя нейроны, - которые нужно выучить довольно мало.

 ![](/images/5bab67340789f49120933392e45e3c0f.png) 
 2D сверточная нейронная сеть

Этот трюк с несколькими копиями одного и того же нейрона примерно аналогичен абстракции функций в математике и информатике. При программировании мы пишем функцию один раз и используем ее во многих местах - если не писать один и тот же код сто раз в разных местах, это ускоряет программирование и приводит к меньшему количеству ошибок. Точно так же сверточная нейронная сеть может однажды выучить нейрон и использовать его во многих местах, упрощая изучение модели и уменьшая количество ошибок.

## Структура сверточных нейронных сетей

Предположим, вы хотите, чтобы нейронная сеть смотрела на аудиосэмплы и предсказывала, говорит человек или нет. Может быть, вы хотите сделать больше анализа, если кто-то говорит.

Вы получаете аудио образцы в разные моменты времени. Образцы расположены равномерно.

 ![](/images/3e84b8472f1a16756cadfd92cc6a1b53.png) 

Самый простой способ попытаться классифицировать их с помощью нейронной сети - просто подключить их все к полностью подключенному слою. Существует множество разных нейронов, и каждый вход соединяется с каждым нейроном.

 ![](/images/5eb9f54084c0376a1d31a82bc8408e55.png) 

Более сложный подход обращает внимание на своего рода _симметрию_ в свойствах, которые полезно искать в данных. Мы очень заботимся о локальных свойствах данных: какая частота звуков присутствует в данное время? Они увеличиваются или уменьшаются? И так далее.

Мы заботимся об одинаковых свойствах во все моменты времени. Полезно знать частоты в начале, полезно знать частоты в середине, а также полезно знать частоты в конце. Опять же, обратите внимание, что это локальные свойства, так как нам нужно только взглянуть на небольшое окно аудиосэмпла, чтобы определить их.

Таким образом, мы можем создать группу нейронов, \\(A\\), которые смотрят на небольшие временные сегменты наших данных. [2](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/#fn2) \\(A\\) просматривает все такие сегменты, вычисляя определенные _функции_. Затем выход этого _сверточного слоя_ подается в полностью связанный слой \\(F\\).

 ![](/images/fbcd03e2376115c47ca4807efc8a0692.png) 

In the above example, \\(A\\) only looked at segments consisting of two points. This isn’t realistic. Usually, a convolution layer’s window would be much larger.

In the following example, \\(A\\) looks at 3 points. That isn’t realistic either – sadly, it’s tricky to visualize \\(A\\) connecting to lots of points.

 ![](/images/8edca52d8fb6135ad47059206dfd2360.png) 

One very nice property of convolutional layers is that they’re composable. You can feed the output of one convolutional layer into another. With each layer, the network can detect higher-level, more abstract features.

In the following example, we have a new group of neurons, \\(B\\) . \\(B\\) is used to create another convolutional layer stacked on top of the previous one.

 ![](/images/3fd1a47ec0ce231670d88ce1d4beb9af.png) 

Convolutional layers are often interweaved with pooling layers. In particular, there is a kind of layer called a max-pooling layer that is extremely popular.

Often, from a high level perspective, we don’t care about the precise point in time a feature is present. If a shift in frequency occurs slightly earlier or later, does it matter?

A max-pooling layer takes the maximum of features over small blocks of a previous layer. The output tells us if a feature was present in a region of the previous layer, but not precisely where.

Max-pooling layers kind of “zoom out”. They allow later convolutional layers to work on larger sections of the data, because a small patch after the pooling layer corresponds to a much larger patch before it. They also make us invariant to some very small transformations of the data.

 ![](/images/1174702401de977b1ddb5a6c684cb49b.png) 

 ![](/images/ca26196a50220824f0fdf928f7b424a8.png) 

In our previous examples, we’ve used 1-dimensional convolutional layers. However, convolutional layers can work on higher-dimensional data as well. In fact, the most famous successes of convolutional neural networks are applying 2D convolutional neural networks to recognizing images.

In a 2-dimensional convolutional layer, instead of looking at segments, \\(A\\) will now look at patches.

For each patch, \\(A\\) will compute features. For example, it might learn to detect the presence of an edge. Or it might learn to detect a texture. Or perhaps a contrast between two colors.

 ![](/images/99e34372a582842947dd5437cc46c26d.png) 

In the previous example, we fed the output of our convolutional layer into a fully-connected layer. But we can also compose two convolutional layers, as we did in the one dimensional case.

 ![](/images/5bab67340789f49120933392e45e3c0f.png) 

We can also do max pooling in two dimensions. Here, we take the maximum of features over a small patch.

What this really boils down to is that, when considering an entire image, we don’t care about the exact position of an edge, down to a pixel. It’s enough to know where it is to within a few pixels.

 ![](/images/9d85a42c7a864c0d810302b7dac8124e.png) 

Three-dimensional convolutional networks are also sometimes used, for data like videos or volumetric data (eg. 3D medical scans). However, they are not very widely used, and much harder to visualize.

Now, we previously said that \\(A\\) was a group of neurons. We should be a bit more precise about this: what is \\(A\\) exactly?

In traditional convolutional layers, \\(A\\) is a bunch of neurons in parallel, that all get the same inputs and compute different features.

For example, in a 2-dimensional convolutional layer, one neuron might detect horizontal edges, another might detect vertical edges, and another might detect green-red color contrasts.

 ![](/images/7efaf64b70dca12e35445b466db5ff1e.png) 

That said, in the recent paper ‘Network in Network’ ( [Lin _et al._ (2013)](http://arxiv.org/abs/1312.4400) ), a new “Mlpconv” layer is proposed. In this model, \\(A\\) would have multiple layers of neurons, with the final layer outputting higher level features for the region. In the paper, the model achieves some very impressive results, setting new state of the art on a number of benchmark datasets.

 ![](/images/2491522bf3ad43cc76f9e04bff0101e5.png) 

That said, for the purposes of this post, we will focus on standard convolutional layers. There’s already enough for us to consider there!

## Results of Convolutional Neural Networks

Earlier, we alluded to recent breakthroughs in computer vision using convolutional neural networks. Before we go on, I’d like to briefly discuss some of these results as motivation.

In 2012, Alex Krizhevsky, Ilya Sutskever, and Geoff Hinton blew existing image classification results out of the water ( [Krizehvsky _et al._ (2012)](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) ).

Their progress was the result of combining together a bunch of different pieces. They used GPUs to train a very large, deep, neural network. They used a new kind of neuron (ReLUs) and a new technique to reduce a problem called ‘overfitting’ (DropOut). They used a very large dataset with lots of image categories ( [ImageNet](http://www.image-net.org/) ). And, of course, it was a convolutional neural network.

Their architecture, illustrated below, was very deep. It has 5 convolutional layers, [3](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/#fn3) with pooling interspersed, and three fully-connected layers. The early layers are split over the two GPUs.

 ![](/images/05bc84347432d78ea35d31d653556023.png) From [Krizehvsky _et al._ (2012)](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) 

They trained their network to classify images into a thousand different categories.

Randomly guessing, one would guess the correct answer 0.1% of the time. Krizhevsky, _et al._ ’s model is able to give the right answer 63% of the time. Further, one of the top 5 answers it gives is right 85% of the time!

 ![](/images/88cd1e0db30c06fb510b8108f8dbbba6.png) Top: 4 correctly classified examples. Bottom: 4 incorrectly classified examples. Each example has an image, followed by its label, followed by the top 5 guesses with probabilities. From [Krizehvsky _et al._ (2012)](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) .

Even some of its errors seem pretty reasonable to me!

We can also examine what the first layer of the network learns to do.

Recall that the convolutional layers were split between the two GPUs. Information doesn’t go back and forth each layer, so the split sides are disconnected in a real way. It turns out that, every time the model is run, the two sides specialize.

 ![](/images/7b7d120378c1c9ce9540c1c16d456ff3.png) Filters learned by the first convolutional layer. The top half corresponds to the layer on one GPU, the bottom on the other. From [Krizehvsky _et al._ (2012)](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) 

Neurons in one side focus on black and white, learning to detect edges of different orientations and sizes. Neurons on the other side specialize on color and texture, detecting color contrasts and patterns. [4](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/#fn4) Remember that the neurons are _randomly_ initialized. No human went and set them to be edge detectors, or to split in this way. It arose simply from training the network to classify images.

These remarkable results (and other exciting results around that time) were only the beginning. They were quickly followed by a lot of other work testing modified approaches and gradually improving the results, or applying them to other areas. And, in addition to the neural networks community, many in the computer vision community have adopted deep convolutional neural networks.

Convolutional neural networks are an essential tool in computer vision and modern pattern recognition.

## Formalizing Convolutional Neural Networks

Consider a 1-dimensional convolutional layer with inputs \\(\\{x\_n\\}\\) and outputs \\(\\{y\_n\\}\\) :

 ![](/images/7be8a90f11666f023ad5fcc55d2a92e7.png) 

It’s relatively easy to describe the outputs in terms of the inputs:

 \\\[y\_n = A(x\_{n}, x\_{n+1}, ...)\\\] 

For example, in the above:

 \\\[y\_0 = A(x\_0, x\_1)\\\]  \\\[y\_1 = A(x\_1, x\_2)\\\] 

Similarly, if we consider a 2-dimensional convolutional layer, with inputs \\(\\{x\_{n,m}\\}\\) and outputs \\(\\{y\_{n,m}\\}\\) :

 ![](/images/e800550c411136d69f5266e16559a2f5.png) 

We can, again, write down the outputs in terms of the inputs:

  \\\[y\_{n,m} = A\\left(\\begin{array}{ccc} x\_{n,\~m}, & x\_{n+1,\~m},& ...,\~\\\\ x\_{n,\~m+1}, & x\_{n+1,\~m+1}, & ..., \~\\\\ &...\\\\\\end{array}\\right)\\\] 

For example:

  \\\[y\_{0,0} = A\\left(\\begin{array}{cc} x\_{0,\~0}, & x\_{1,\~0},\~\\\\ x\_{0,\~1}, & x\_{1,\~1}\~\\\\\\end{array}\\right)\\\]  \\\[y\_{1,0} = A\\left(\\begin{array}{cc} x\_{1,\~0}, & x\_{2,\~0},\~\\\\ x\_{1,\~1}, & x\_{2,\~1}\~\\\\\\end{array}\\right)\\\] 

If one combines this with the equation for \\(A(x)\\) ,

 \\\[A(x) = \\sigma(Wx + b)\\\] 

one has everything they need to implement a convolutional neural network, at least in theory.

In practice, this is often not best way to think about convolutional neural networks. There is an alternative formulation, in terms of a mathematical operation called _convolution_ , that is often more helpful.

The convolution operation is a powerful tool. In mathematics, it comes up in diverse contexts, ranging from the study of partial differential equations to probability theory. In part because of its role in PDEs, convolution is very important in the physical sciences. It also has an important role in many applied areas, like computer graphics and signal processing.

For us, convolution will provide a number of benefits. Firstly, it will allow us to create much more efficient implementations of convolutional layers than the naive perspective might suggest. Secondly, it will remove a lot of messiness from our formulation, handling all the bookkeeping presently showing up in the indexing of \\(x\\) s – the present formulation may not seem messy yet, but that’s only because we haven’t got into the tricky cases yet. Finally, convolution will give us a significantly different perspective for reasoning about convolutional layers.

> I admire the elegance of your method of computation; it must be nice to ride through these fields upon the horse of true mathematics while the like of us have to make our way laboriously on foot.  — Albert Einstein

## Next Posts in this Series

 [ **Read the next post!** ](https://colah.github.io/posts/2014-07-Understanding-Convolutions/) 

This post is part of a series on convolutional neural networks and their generalizations. The first two posts will be review for those familiar with deep learning, while later ones should be of interest to everyone. To get updates, subscribe to my [RSS feed](https://colah.github.io/rss.xml) !

Please comment below or on the side. Pull requests can be made on [github](https://github.com/colah/Conv-Nets-Series) .

## Acknowledgments

I’m grateful to Eliana Lorch, Aaron Courville, and Sebastian Zany for their comments and support.

* * *

1.  It should be noted that not all neural networks that use multiple copies of the same neuron are convolutional neural networks. Convolutional neural networks are just one type of neural network that uses the more general trick, _weight-tying_ . Other kinds of neural network that do this are recurrent neural networks and recursive neural networks. [↩](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/#fnref1) 
    
2.  Groups of neurons, like \\(A\\) , that appear in multiple places are sometimes called _modules_ , and networks that use them are sometimes called _modular neural networks_ . [↩](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/#fnref2) 
    
3.  They also test using 7 in the paper. [↩](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/#fnref3) 
    
4.  This seems to have interesting analogies to rods and cones in the retina. [↩](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/#fnref4)



**********
[Свёрточная нейронная сеть](/tags/%D0%A1%D0%B2%D1%91%D1%80%D1%82%D0%BE%D1%87%D0%BD%D0%B0%D1%8F%20%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F%20%D1%81%D0%B5%D1%82%D1%8C.md)
[CNN](/tags/CNN.md)
[Convolutional Neural Networks](/tags/Convolutional%20Neural%20Networks.md)
