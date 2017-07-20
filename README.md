# multilabel-MXNet
This is the implement of the multilabel image classificaton in MXNet. Multilabel means each image belong to 1 or more than 1 labels and it is different from multi task.

This implement doesn't need recompile MXNet and is very convenient for you to use. Firstly, I assume that you can use MXNet normally. Then, do as follows:

1. If you are doing a single label image classification, your .lst file may like this(take 4 classes as example):

5247	0.000000	image1.jpg

33986	1.000000	image2.jpg

39829	2.000000	image3.jpg

15647	3.000000	image4.jpg

10369	1.000000	image5.jpg

22408	3.000000	image6.jpg

2598	2.000000	image7.jpg

For multilabel image classification, you should create .lst file as this(take 8 classes as example):

5247	1.000000	0.000000	0.000000	0.000000	1.000000	0.000000	0.000000	0.000000	image1.jpg

33986	1.000000	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	image2.jpg

39829	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	image3.jpg

15647	1.000000	0.000000	0.000000	1.000000	0.000000	0.000000	0.000000	0.000000	image4.jpg

10369	0.000000	0.000000	1.000000	0.000000	0.000000	0.000000	1.000000	1.000000	image5.jpg

22408	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	image6.jpg

2598	0.000000	1.000000	1.000000	0.000000	1.000000	1.000000	0.000000	0.000000	image7.jpg

in this implement, we only use .lst and raw image as the input instead of .rec file.

2. Putting the script fine-tune-multilabel.py into ~/mxnet/example/image-classification/. Putting scripts fit_multilabel.py and multilabel_data.py into ~/mxnet/example/image-classification/common/. ~/mxnet is your mxnet project which you can clone from https://github.com/dmlc/mxnet.

3. class_train_multilabel.sh is the train script for you to start fine-tune quickly. You should open this script and change the path of fine-tune-multilabel.py, .lst file, imagefile and model-prefix after you clone the project.

   Then run: 
   sh class_train_multilabel.sh

#################################### More details ##############################################

1. fine-tune-multilabel.py is modified from fine-tune.py which you can find from https://github.com/dmlc/mxnet/blob/master/example/image-classification/fine-tune.py. There are something different:

1.1 I add two classes CrossEntropyLoss and class CrossEntropyProb, these two classes can be found from https://github.com/dmlc/mxnet/blob/master/example/recommenders/crossentropy.py. There is a bug in crossentropy.py: grad = -1. / (p - self.eps_1 + y). The (p - self.eps_1 + y) part will be 0 in some cases, so I use d_new = p - self.eps_1 + y, d_new[d_new==0] = self.eps_1,   grad = -1. / d_new instead.

1.2 In the get_fine_tune_model function, I use net = mx.symbol.sigmoid(data=net, name='sig') and net = mx.symbol.Custom(data=net,name='softmax', op_type='CrossEntropyLoss') instead of mx.symbol.softmaxOutput to deal with multilabel problem.

2. fit_multilabel.py is modified from fit.py which you can find from https://github.com/dmlc/mxnet/blob/master/example/image-classification/common/fit.py. There are something different:

2.1 Define a new function: ml_acc, which is used for calculate the "accuracy" of multilabel. So in model.fit(), eval_metric = mx.metric.np(ml_acc) instead of eval_metric=eval_metrics. You can also define a new accuracy function if you need, ml_acc is just an example.

3. multilabel_data.py is modified from data.py which you can find from https://github.com/dmlc/mxnet/blob/master/example/image-classification/common/data.py. The original data.py is used for reading .rec file by using mx.io.ImageRecordIter class, but my multilabel_data.py can read from .lst and raw images by using mx.image.ImageIter class. You can also use data.py if you are familiar with it and it will not affect multilabel. You need only modify fine-tune-multilabel.py where you use multilabel_data.py into data.py. For example: if you use multilabel_data.py you should :from common import multilabel_data, but if you use data.py, you should from common import data. You should change all the multilabel_data into data in fine-tune-multilabel.py if you still want to use data.py to read .rec file.
