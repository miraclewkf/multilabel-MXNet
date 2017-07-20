import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet
from common import multilabel_data, fit_multilabel, modelzoo
import mxnet as mx
import numpy as np

class CrossEntropyLoss(mx.operator.CustomOp):
    """An output layer that calculates gradient for cross-entropy loss
    y * log(p) + (1-y) * log(p)
    for label "y" and prediction "p".
    However, the output of this layer is the original prediction -- same as 
    the "data" input, making it useful for tasks like "predict".
    If you actually want to use the calculated loss, see CrossEntropyLoss op.

    This is useful for multi-label prediction where each possible output
    label is considered independently.
    Cross-entropy loss provides a very large penalty for guessing 
    the wrong answer (0 or 1) confidently.
    The gradient calculation is optimized for y only being 0 or 1.
    """

    eps = 1e-6 # Avoid -inf when taking log(0)
    eps1 = 1. + eps
    eps_1 = 1. - eps

    def forward(self, is_train, req, in_data, out_data, aux):
        # Shapes:
        #  b = minibatch size
        #  d = number of dimensions
        actually_calculate_loss = False
        if actually_calculate_loss:
            p = in_data[0].asnumpy()  # shape=(b,d)
            y = in_data[1].asnumpy()
            out = y * np.log(p+self.eps) + (1.-y) * np.log((self.eps1) - p)
            self.assign(out_data[0], req[0], mx.nd.array(out))
        else:
            # Just copy the predictions forward
            self.assign(out_data[0], req[0], in_data[0])


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.approx_backward(req, out_grad, in_data, out_data, in_grad, aux)
        #self.exact_backward(req, out_grad, in_data, out_data, in_grad, aux)

    def approx_backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Correct grad = (y-p)/(p-p^2)
        But if y is just 1 or 0, then this simplifies to
        grad = 1/(p-1+y)
        which is more numerically stable
        """
        p = in_data[0].asnumpy()  # shape=(b,d)
        y = in_data[1].asnumpy()
        d_new = p - self.eps_1 + y
        d_new[d_new==0] = self.eps_1
        grad = -1. / d_new
        self.assign(in_grad[0], req[0], mx.nd.array(grad))


    def exact_backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """grad = (y-p)/(p-p^2)
        """
        p = in_data[0].asnumpy()  # shape=(b,d)
        y = in_data[1].asnumpy()  # seems right
        grad = (p - y) / ((p+self.eps) * (self.eps1 - p))
        self.assign(in_grad[0], req[0], mx.nd.array(grad))


@mx.operator.register("CrossEntropyLoss")
class CrossEntropyProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CrossEntropyProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data','label']

    def list_outputs(self):
        return ['preds']

    def create_operator(self, ctx, shapes, dtypes):
        return CrossEntropyLoss()

    def infer_shape(self, in_shape):
        if in_shape[0] != in_shape[1]:
            raise ValueError("Input shapes differ. data:%s. label:%s. must be same"
                    % (str(in_shape[0]),str(in_shape[1])))
        output_shape = in_shape[0]
        return in_shape, [output_shape], []
        
def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc',lr_mult=5)
    net = mx.symbol.sigmoid(data=net, name='sig')
    net = mx.symbol.Custom(data=net,name='softmax', op_type='CrossEntropyLoss')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit_multilabel.add_fit_args(parser)
    multilabel_data.add_data_args(parser)
    aug = multilabel_data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    # use less augmentations for fine-tune
    multilabel_data.set_data_aug_level(parser, 1)
    # use a small learning rate and less regularizations
    parser.set_defaults(image_shape='3,224,224', num_epochs=8,
                        lr=.01, lr_step_epochs='5', wd=0, mom=0)

    args = parser.parse_args()

    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    (prefix, epoch) = modelzoo.download_model(
        args.pretrained_model, os.path.join(dir_path, 'model'))
    if prefix is None:
        (prefix, epoch) = (args.pretrained_model, args.load_epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, args.layer_before_fullc)

    # train
    
    fit_multilabel.fit(args        = args,
            network     = new_sym,
            data_loader = multilabel_data.get_lst_iter,
            arg_params  = new_args,
            aux_params  = aux_params)
