python /mxnet/example/image-classification/fine-tune-multilabel.py \
--data-train your/path/to/lstfile/multilabel_train.lst --image-train your/path/to/images \
--data-val your/path/to/lstfile/multilabel_test.lst --image-val your/path/to/images \
--gpus 0 --batch-size 32 --lr 0.001 --num-classes 8 --num-epochs 15 --num-example 500000 --lr-step-epochs 15 \
--pretrained-model imagenet1k-resnet-50 --load-epoch 0000 \
--model-prefix your/path/to/save/model/and/log/multilabel-MXNet/multilabel-resnet50
