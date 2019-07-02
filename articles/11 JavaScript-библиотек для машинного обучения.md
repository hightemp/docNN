Javascript-библиотеки используют для определения, обучения и запуска моделей глубокого обучения, визуализации данных полностью в браузере. Они значительно облегчают жизнь разработчику. Ниже представлены изящные библиотеки, которые объединяют Javascript, [машинное обучение](https://neurohive.io/ru/osnovy-data-science/vvedenie-v-mashinnoe-obuchenie-kto-ego-primenjaet-i-kak-stat-razrabotchikom/ "машинное обучение") , глубокие нейронные сети и даже NLP.

## 1\. Brain.js

 [Brain.js](https://github.com/BrainJS/brain.js) — Javascript библиотека для искусственных нейронных сетей, заменяющая «мозговую» библиотеку, которая предлагает разные  [типы](https://github.com/BrainJS/brain.js#neural-network-types) сетей в зависимости от конкретных задач. Используется с [Node.js](https://www.npmjs.com/package/brainjs) или в [браузере.](https://raw.githubusercontent.com/harthur-org/brain.js/master/browser.js)  Здесь представлено [демо](https://harthur.github.io/brain/) тренировки сети для распознавания цветовых контрастов.

 ![Brain.js - Javascript библиотека для искусственных нейронных сетей](/images/9bdd32fe5eb8ff68c2d1644868eae308.png) 

Обучение Brain.js распознавать цветовые контрасты

## 2\. Synaptic

Synaptic — Javascript [библиотека](http://caza.la/synaptic/#/) для нейронных сетей для node.js и браузера, которая позволяет обучать архитектуры нейронных сетей первого и второго порядков. Проект содержит несколько встроенных архитектур  — многослойный перцептрон, многослойная сеть долгой краткосрочной памяти, LSM (liquid state machine) и тренер (trainer), способный обучать сети.

 ![Synaptic - Javascript библиотека](/images/4a329c551041798253e6dc5290f75235.png) 

Фильтрация изображения с помощью перцептрона Synaptic

## 3\. Neataptic

Эта [библиотека](https://wagenaartje.github.io/neataptic/) предоставляет возможность быстро осуществлять нейроэволюцию и [обратное распространение](https://neurohive.io/ru/osnovy-data-science/obratnoe-rasprostranenie/ "обратное распространение") для браузера и Node.js. Библиотека содержит несколько встроенных сетей — перцептрон, LSTM, GRU, Nark и другие. Для новичков [есть туториал](https://wagenaartje.github.io/neataptic/docs/tutorials/training/) , помогающий реализовать тренировку сети.

 ![Neaptic библиотека JS](/images/8aa635e11e6c5fe8886989217147c657.png) 

Демо поиска цели на Neaptic

## 4\. Conventjs

Эта популярная [библиотека](https://github.com/karpathy/convnetjs) , разработанная PhD студентом из Стэнфорда Андреем Карпатым, который сейчас работает в Tesla. Хотя она не поддерживается последние 4 года, Conventjs остается одним из самых интересных проектов в этом списке. Conventjs представляет из себя написанную на Javascript реализацию нейронных сетей, поддерживающую  распространенные модули — классификацию, регрессию, экспериментальный модуль обучения с подкреплением. С помощью этой библиотеки можно даже обучать [сверточную нейросеть](https://neurohive.io/vidy-nejrosetej/glubokaya-svertochnaja-nejronnaja-set/ "сверточную нейросеть") для обработки изображений.

 ![javascript библиотека машинного обучения Convent.js](/images/3af9f23c5cf109080b34e4c712ab4fce.png) 

Задача двумерной классификации при помощи двухслойной нейросети на Convent.js

## 5\. Webdnn

 [Webdnn](https://mil-tokyo.github.io/webdnn/) — японская библиотека, созданная для быстрой работы с предобученными глубокими нейросетевыми моделями в браузере. Хотя запуск DNN (Глубокой нейронная сети) в браузере требует больших вычислительных ресурсов, этот фреймворк оптимизирует DNN модель так, что данные модели сжимаются, а исполнение ускоряется при помощи JavascriptAPI, таких как WebAssembly и WebGPU.

 ![японская библиотека JS для машинного обучения](/images/e9b4098b4238dc5cfd14acc770a04af7.png) 

Пример трансфера стиля

## 6\. Tensorflow.js

 [Библиотека от Google](https://js.tensorflow.org/) (преемница популярной deeplearnjs) дает возможность обучать нейронные сети в браузере или запускать предобученные модели в режиме вывода. Создатели библиотеки утверждают, что она может быть использована как [NumPy](http://www.numpy.org/) для веба. [Tensorflow.js](https://neurohive.io/ru/tutorial/obzor-tensorflow-js-mashinnoe-obuchenie-na-javascript/) с простым в работе API может быть использована в разнообразных полезных приложениях. Библиотека также активно поддерживается.

### 7\. TensorFlow Deep Playground

Deep playground  — [инструмент](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.08014&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) для интерактивной визуализации нейронных сетей, написанный на TypeScript с использованием d3.js. Хотя этот проект в основном содержит самую базовую площадку для [tensorflow](https://neurohive.io/ru/tutorial/tensorflow-tutorial-tenzory-i-vektory/ "tensorflow") , он может быть использован для различных целей, например, в качестве очень выразительного обучающего инструмента.

 ![Tensorflow библиотке Javascript](/images/055710ca9b8ec096617182883ba3ec84.png) 

Игровая площадка Tensorflow

### 8\. Compromise

Compromise — популярная [библиотека](http://compromise.cool/) , которая позволяет осуществлять обработку естественного языка (NLP) при помощи Javascript. Она базовая, компилируется в единственный маленький файл. По некоторым причинам, скромного функционала вполне хватает для того, чтобы сделать Compromise главным кандидатом для использования практически в любом приложении, в котором требуется базовый NLP.

 ![Compromise](/images/1e15637b1679e8dd52675f28b37cf1fe.png) 

Compromise напоминает, как в действительности выглядит английский

### 9\. Neuro.js

Этот [проект](https://github.com/janhuenermann/neurojs) представляет собой Javascript библиотеку глубокого обучения и обучения с подкреплением в браузере. Из-за реализации полнофункционального фреймворка машинного обучения на основе нейронных сетей с поддержкой обучения с подкреплением, Neuro.js считается преемником Conventjs.

 ![neurojs library Javascript machine learning models](/images/36e69c52d80739874aa0364f68f6b7df.png) 

Беспилотное авто с Neuro.js

### 10\. mljs

Это группа репозиториев, содержащая [инструменты](https://github.com/mljs/ml) для машинного обучения для Javascript, разработана группой mljs. Mljs включает в себя [обучение с учителем](https://neurohive.io/ru/osnovy-data-science/obuchenie-s-uchitelem-bez-uchitelja-s-podkrepleniem/ "обучение с учителем") и без, искусственные нейронные сети, алгоритмы регрессии и поддерживает библиотеки для статистики, математики тому подобное. Здесь можно найти краткое [введение](https://hackernoon.com/machine-learning-with-javascript-part-1-9b97f3ed4fe5) в тему.

 ![репозитории для машинного обучения](/images/fe1b9dd442e6e20901ebcdb465c21dc0.png) 

Проект mljs на GitHub

### 11\. Mind

Mind — гибкая нейросетевая [библиотека](http://stevenmiller888.github.io/mindjs.net/) для Node.js и браузера. Mind учится предсказывать, выполняя матричную обработку тренировочных данных и обеспечивая настраиваемую топологию сети. Можете использовать уже существующие разработки, что может быть весьма полезно для ваших приложений.

 ![гибкая нейросетевая библиотека для Node.js и браузера](/images/2ea1e683aaf332567925ab35269e9920.png) 

## Достойны упоминания:

### Natural

Активно поддерживаемая [библиотека](https://github.com/NaturalNode/natural) для Node.js, которая обеспечивает: токенизацию, стемминг (сокращение слова до необязательно морфологического корня), классификацию, фонетику, tf-idf, WordNet и другое.

### Incubator-mxnet

MXnet от Apache — фреймворк глубокого обучения, который позволяет на лету [смешивать символьное и императивное программирование](https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts) со слоем оптимизации графа для достижения результата. [MXnet.js](https://github.com/dmlc/mxnet.js/) обеспечивает API для глубокого обучения в браузере.

### Keras JS

Эта [библиотека](https://transcranial.github.io/keras-js/#/) запускает модели Keras в браузере с поддержкой GPU при помощи технологии WebGL.  Так как Keras использует в качестве бэк-енда различные фреймворки, модели могут быть обучены в TensorFlow, CNTK, а также и в других фреймворках.

### Deepforge

Deepforge — [среда разработки](http://deepforge.org/) для глубокого обучения, которая позволяет быстро конструировать архитектуру [нейронной сети](https://neurohive.io/ru/osnovy-data-science/osnovy-nejronnyh-setej-algoritmy-obuchenie-funkcii-aktivacii-i-poteri/ "нейронной сети") и пайплайны машинного обучения. В Deepforge содержится также встроенная функция контроля версий для воспроизведения экспериментов. Сюда стоит заглянуть.

### Land Lines

Land Lines — не столько библиотека, сколько очень [занимательная веб-игра](https://lines.chromeexperiments.com/) на основе эксперимента Chrome от Google. Нельзя сказать, для чего нужна эта штука, но она позабавит хотя бы 15 минут.

**********
[javascript](/tags/javascript.md)
