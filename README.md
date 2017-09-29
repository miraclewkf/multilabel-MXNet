# multilabel-MXNet
This is the implement of the multilabel image classificaton in MXNet. Multilabel means each image belong to 1 or more than 1 labels and it is different from multi task.

This implement doesn't need recompile MXNet and is very convenient for you to use. Firstly, I assume that you can use MXNet normally. Then, do as follows:

 1. If you are doing a single label image classification, your .lst file may like this(take 4 classes as example):

|ID	|label   |      image_name|
|:------|:-------|:---------------| 
|5247	|0.000000|	image1.jpg|
|33986	|1.000000|	image2.jpg|
|39829	|2.000000|	image3.jpg|
|15647	|3.000000|	image4.jpg|
|10369	|1.000000|	image5.jpg|
|22408	|3.000000|	image6.jpg|
|2598	|2.000000|	image7.jpg|

For multilabel image classification, you should create .lst file as this(take 8 classes as example):

| ID  | LABEL     | IMAGE_NAME|
|:---:|:---------------------------------------------------------------------------------------------:|:---------:|
|5247 |	1.000000 |	1.000000|	1.000000|	0.000000|	0.000000|	0.000000|	0.000000|	0.000000|	image1.jpg|


in this implement, we only use .lst and raw image as the input instead of .rec file.



2. run_train.sh is the train script for you to start fine-tune quickly. You should open this script and change the path of train_multilabel.py, .lst file, imagefile and model-prefix after you clone the project.

   Then run: 
   sh run_train.sh

## More details

1. --num-classes 8 in run_train.sh means the maximum number of label is 8 classes. For example, iamge1 has label 1,5; image2 has label 1,2,3; image3 has label 1. You can change the number for your data.

2. train_multilabel.py is 

2.1. In the get_fine_tune_model function, I use net = mx.symbol.sigmoid(data=net, name='sig') and net = mx.symbol.Custom(data=net,name='softmax', op_type='CrossEntropyLoss') instead of mx.symbol.softmaxOutput to deal with multilabel problem. sigmoid layer take the output of full connection as input ahd translate it into (0,1), which means the probability. CrossEntropy layer take the probability (0,1) as input and calculate the loss.


