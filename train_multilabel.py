import argparse
import mxnet as mx
import os, sys
import numpy as np
sys.path.insert(0, "./settings")
sys.path.insert(0, "../")

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

from crossentropy import *

def get_fine_tune_model(sym, arg_params, num_classes, layer_name):
    
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.sigmoid(data=net, name='sig')
    net = mx.symbol.Custom(data=net, name='softmax', op_type='CrossEntropyLoss')

    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)

def multi_factor_scheduler(begin_epoch, epoch_size, step=[5,10], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

def train_model(model, gpus, epoch=0, num_epoch=20, kv='device', num_class=6):
    train = mx.image.ImageIter(
        batch_size          = args.batch_size,
        data_shape          = (3,224,224),        
        label_width         = num_class,
        path_imglist        = args.data_train,
        path_root           = args.image_train,
        part_index          = kv.rank,
        num_parts           = kv.num_workers,
        shuffle             = True,
        data_name           = 'data',
        label_name          = 'softmax_label',
        aug_list            = mx.image.CreateAugmenter((3,224,224),resize=224,rand_crop=True,rand_mirror=True,mean=True,std=True))

    val = mx.image.ImageIter(
        batch_size          = args.batch_size,
        data_shape          = (3,224,224),
        label_width         = num_class,
        path_imglist        = args.data_val,
        path_root           = args.image_val,
        part_index          = kv.rank,
        num_parts           = kv.num_workers,       
        data_name           = 'data',
        label_name          = 'softmax_label',
        aug_list            = mx.image.CreateAugmenter((3,224,224),resize=224,mean=True,std=True))

    kv = mx.kvstore.create(args.kv_store)

    prefix = model
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, 'flatten0')

    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    lr_scheduler=multi_factor_scheduler(args.epoch, epoch_size)

    optimizer_params = {
            'learning_rate': args.lr,
            'momentum' : args.mom,
            'wd' : args.wd,
            'lr_scheduler': lr_scheduler}
    initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]
        
    model = mx.mod.Module(
        context       = devs,
        symbol        = new_sym
    )

    checkpoint = mx.callback.do_checkpoint(args.save_result+args.save_name)

    def acc(label, pred, label_width = num_class):
        return float((label == np.round(pred)).sum()) / label_width / pred.shape[0]

    def loss(label, pred):
        loss_all = 0
        for i in range(len(pred)):
            loss = 0
            loss -= label[i] * np.log(pred[i] + 1e-6) + (1.- label[i]) * np.log(1. + 1e-6 - pred[i])
            loss_all += np.sum(loss)
        loss_all = float(loss_all)/float(len(pred) + 0.000001)
        return  loss_all


    eval_metric = list()
    eval_metric.append(mx.metric.np(acc))
    eval_metric.append(mx.metric.np(loss))

    model.fit(train,
              begin_epoch=epoch,
              num_epoch=num_epoch,
              eval_data=val,
              eval_metric=eval_metric,
              validation_metric=eval_metric,
              kvstore=kv,
              optimizer='sgd',
              optimizer_params=optimizer_params,
              arg_params=new_args,
              aux_params=aux_params,
              initializer=initializer,
              allow_missing=True,
              batch_end_callback=mx.callback.Speedometer(args.batch_size, 20),
              epoch_end_callback=checkpoint)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--model',         type=str, required=True,)
    parser.add_argument('--gpus',          type=str, default='0')
    parser.add_argument('--batch-size',    type=int, default=200)
    parser.add_argument('--epoch',         type=int, default=0)
    parser.add_argument('--image-shape',   type=str, default='3,224,224')
    parser.add_argument('--data-train',    type=str)
    parser.add_argument('--image-train',   type=str)
    parser.add_argument('--data-val',      type=str)
    parser.add_argument('--image-val',     type=str)
    parser.add_argument('--num-classes',   type=int, default=6)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--num-epoch',     type=int, default=2)
    parser.add_argument('--kv-store',      type=str, default='device', help='the kvstore type')
    parser.add_argument('--save-result',   type=str, help='the save path')
    parser.add_argument('--num-examples',  type=int, default=20000)
    parser.add_argument('--mom',           type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd',            type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--save-name',     type=str, help='the save name of model')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    kv = mx.kvstore.create(args.kv_store)

    if not os.path.exists(args.save_result):
        os.mkdir(args.save_result)
    hdlr = logging.FileHandler(args.save_result+ '/train.log')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logging.info(args)

    train_model(model=args.model, gpus=args.gpus, epoch=args.epoch, num_epoch=args.num_epoch, kv=kv, num_class=args.num_classes)

