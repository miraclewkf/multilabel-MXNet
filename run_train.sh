python train_multilabel.py --epoch 0 --model your/pretrained_mode/imagenet1k-resnet-50 --batch-size 64 --num-classes 8 \
--data-train your/path/to/lstfile/multilabel_train.lst --image-train your/path/to/images \
--data-val your/path/to/lstfile/multilabel_test.lst --image-val your/path/to/images \
--num-examples 100000 --lr 0.001 --gpus 0 --num-epoch 15 --save-result your/path/to/save/model/ --save-name multilabel-resnet50
