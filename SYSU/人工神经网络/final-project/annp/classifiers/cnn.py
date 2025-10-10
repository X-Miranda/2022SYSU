from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *

import numpy as np

import numpy as np

class MyConvNet:
    def __init__(self, input_dim=(3, 32, 32), num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        C, H, W = input_dim
        F1, F2 = 32, 64  # 卷积层滤波器数量
        HH, WW = 7, 7    # 使用3x3卷积核
        stride = 1
        pad = (HH - 1) // 2
        
        # 初始化权重 - 使用更小的weight_scale(1e-3)
        self.params['W1'] = np.random.normal(0, weight_scale, (F1, C, HH, WW))
        self.params['b1'] = np.zeros(F1)
        self.params['W2'] = np.random.normal(0, weight_scale, (F2, F1, HH, WW))
        self.params['b2'] = np.zeros(F2)
        
        # 计算卷积层输出尺寸
        conv1_out_h = 1 + (H + 2 * pad - HH) // stride
        conv1_out_w = 1 + (W + 2 * pad - WW) // stride
        pool1_out_h = conv1_out_h // 2
        pool1_out_w = conv1_out_w // 2
        
        conv2_out_h = 1 + (pool1_out_h + 2 * pad - HH) // stride
        conv2_out_w = 1 + (pool1_out_w + 2 * pad - WW) // stride
        pool2_out_h = conv2_out_h // 2
        pool2_out_w = conv2_out_w // 2
        
        # 打印卷积层和池化层的输出维度，确保它们是正确的
        print("Conv1 output shape:", (F1, conv1_out_h, conv1_out_w))
        print("Pool1 output shape:", (F1, pool1_out_h, pool1_out_w))
        print("Conv2 output shape:", (F2, conv2_out_h, conv2_out_w))
        print("Pool2 output shape:", (F2, pool2_out_h, pool2_out_w))
        
        # 全连接层
        hidden_dim = 500  
        self.params['W3'] = np.random.normal(0, weight_scale, 
                                           (F2 * pool2_out_h * pool2_out_w, hidden_dim))
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['W4'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b4'] = np.zeros(num_classes)
        
        # 转换为正确的数据类型
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'
        
        conv_param = {'stride': 1, 'pad': (7 - 1) // 2}  # 3x3卷积的padding
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        
        # 第一层: Conv -> ReLU -> Pool
        conv1, cache1 = conv_forward_fast(X, W1, b1, conv_param)
        relu1, cache_relu1 = relu_forward(conv1)
        pool1, cache_pool1 = max_pool_forward_fast(relu1, pool_param)
        
        # 第二层: Conv -> ReLU -> Pool
        conv2, cache2 = conv_forward_fast(pool1, W2, b2, conv_param)
        relu2, cache_relu2 = relu_forward(conv2)
        pool2, cache_pool2 = max_pool_forward_fast(relu2, pool_param)
        
        # 全连接层1: Affine -> ReLU
        fc1_input = pool2.reshape(pool2.shape[0], -1)
        fc1, cache_fc1 = affine_forward(fc1_input, W3, b3)
        relu3, cache_relu3 = relu_forward(fc1)
        
        # 全连接层2: Affine
        scores, cache_fc2 = affine_forward(relu3, W4, b4)
        
        if y is None:
            return scores
        
        loss, grads = 0, {}
        
        # 计算损失和梯度
        loss, dscores = softmax_loss(scores, y)
        # 添加L2正则化
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
        
        # 反向传播
        dfc2, grads['W4'], grads['b4'] = affine_backward(dscores, cache_fc2)
        drelu3 = relu_backward(dfc2, cache_relu3)
        dpool2, grads['W3'], grads['b3'] = affine_backward(drelu3, cache_fc1)
        
        dpool2 = dpool2.reshape(pool2.shape)
        drelu2 = max_pool_backward_fast(dpool2, cache_pool2)
        dconv2 = relu_backward(drelu2, cache_relu2)
        dpool1, grads['W2'], grads['b2'] = conv_backward_fast(dconv2, cache2)
        
        drelu1 = max_pool_backward_fast(dpool1, cache_pool1)
        dconv1 = relu_backward(drelu1, cache_relu1)
        _, grads['W1'], grads['b1'] = conv_backward_fast(dconv1, cache1)
        
        # 添加正则化梯度
        grads['W4'] += self.reg * W4
        grads['W3'] += self.reg * W3
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1
        
        return loss, grads
    
    
class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        pooling='maxpool',
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        - pooling: What type of pooling the network should use. Valid values are
          "maxpool" or "avgpool"
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.pooling = pooling

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        C, H, W = input_dim
        
        # Initialize weights for convolutional layer
        self.params['W1'] = weight_scale * np.random.randn(
            num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        
        # Calculate dimensions after conv and pooling
        conv_H = 1 + (H + 2 * (filter_size // 2) - filter_size) // 1  # assuming stride=1
        conv_W = 1 + (W + 2 * (filter_size // 2) - filter_size) // 1
        pool_H = conv_H // 2  # assuming 2x2 pooling with stride=2
        pool_W = conv_W // 2
        
        # Initialize weights for affine layers
        self.params['W2'] = weight_scale * np.random.randn(
            num_filters * pool_H * pool_W, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(
            hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in annp/fast_layers.py and  #
        # annp/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # Forward pass
        # Conv -> ReLU -> Pool
        conv_out, conv_cache = conv_forward_fast(X, W1, b1, conv_param)
        relu1_out, relu1_cache = relu_forward(conv_out)
        # 使用快速池化前向传播
        pool_out, pool_cache = max_pool_forward_fast(relu1_out, pool_param)
        
        # Affine -> ReLU
        N = X.shape[0]
        pool_out_flat = pool_out.reshape(N, -1)
        affine1_out, affine1_cache = affine_forward(pool_out_flat, W2, b2)
        relu2_out, relu2_cache = relu_forward(affine1_out)
        
        # Final affine
        scores, affine2_cache = affine_forward(relu2_out, W3, b3)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass
        data_loss, dscores = softmax_loss(scores, y)
        
        # Final affine backward
        drelu2_out, dW3, db3 = affine_backward(dscores, affine2_cache)
        
        # Affine -> ReLU backward
        daffine1_out = relu_backward(drelu2_out, relu2_cache)
        dpool_out_flat, dW2, db2 = affine_backward(daffine1_out, affine1_cache)
        
        # Reshape pooled gradients
        dpool_out = dpool_out_flat.reshape(pool_out.shape)
        
        # Pool backward
        drelu1_out = max_pool_backward_fast(dpool_out, pool_cache)
        
        # ReLU backward
        dconv_out = relu_backward(drelu1_out, relu1_cache)
        
        # Conv backward
        dX, dW1, db1 = conv_backward_fast(dconv_out, conv_cache)
        
        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        
        reg_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        loss = data_loss + reg_loss
        
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
