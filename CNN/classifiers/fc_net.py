import numpy as np

from layers import*
from layer_utils import*


class FullyConnectedNet(object):
    """
    affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        for i in range(len(hidden_dims)):
            layer = hidden_dims[i]
            self.params['W'+str(i+1)] = np.random.normal(0, weight_scale, size=(input_dim, layer))
            self.params['b'+str(i+1)] = np.zeros(layer)
            if self.use_batchnorm:
                self.params['gamma'+str(i+1)] = np.ones(layer)
                # gamma 初始為 1
                self.params['beta'+str(i+1)] = np.zeros(layer)
            input_dim = layer
            # 一層接一層
        self.params['W'+str(self.num_layers)] = np.random.normal(0, weight_scale, size=(layer, num_classes))
        self.params['b'+str(self.num_layers)] = np.zeros(num_classes)

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
                
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None

        hidden = X
        cache = {}
        drop_cache = {}
        for i in range(self.num_layers - 1):
            # 先 batchnorm 後 dropout
            if self.use_batchnorm:
                hidden, cache[i] = affine_bn_relu_forward(hidden, 
                    self.params['W'+str(i+1)], self.params['b'+str(i+1)], 
                    self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], 
                    self.bn_params[i])
            else:
                hidden, cache[i] = affine_relu_forward(hidden, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            if self.use_dropout:
                hidden, drop_cache[i] = dropout_forward(hidden, self.dropout_param)
        out, cache[self.num_layers - 1] = affine_forward(hidden, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])
        scores = out

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dout = softmax_loss(out, y)
        dhidden, dw, db = affine_backward(dout, cache[self.num_layers-1])
        loss +=  0.5 * self.reg * np.sum(self.params['W'+str(self.num_layers)] * self.params['W'+str(self.num_layers)])
        grads['W'+str(self.num_layers)] = dw + self.reg * self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db

        
        for i in range(self.num_layers - 1, 0, -1):
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(i)] * self.params['W'+str(i)])
            if self.use_dropout:
                dhidden = dropout_backward(dhidden, drop_cache[i-1])
            if self.use_batchnorm:
                dhidden, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhidden, cache[i-1])
                grads['gamma'+str(i)] = dgamma
                grads['beta'+str(i)] = dbeta
            else:
                dhidden, dw, db = affine_relu_backward(dhidden, cache[i-1])
            
            grads['W'+str(i)] = dw + self.reg * self.params['W'+str(i)]
            grads['b'+str(i)] = db


        return loss, grads
