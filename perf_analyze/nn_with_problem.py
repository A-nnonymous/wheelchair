---------------------------------------rotate_ppyoloe_r_ppyoloe_r_crn_s_3x_dota-SIR_181---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[96, 96, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 96, 128, 128], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 96, 1, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_1, bias=self.parameter_0, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_3 = paddle.tensor.ops.sigmoid(var_2)
        var_4 = var_0.__mul__(var_3)
        return var_4



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 96, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 96, 128, 128], dtype=paddle.float32),
        paddle.rand(shape=[1, 96, 1, 1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 96, 128, 128]).astype('float32'),
        np.random.random(size=[1, 96, 1, 1]).astype('float32'),
    )
    return inputs


---------------------------------------yolov3_yolov3_mobilenet_v3_large_270e_coco-SIR_31---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[120],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[30],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[30, 120, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[120, 30, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 120, 68, 68], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_0, output_size=1, data_format='NCHW', name=None)
        var_2 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_2, bias=self.parameter_1, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_3 = paddle.nn.functional.activation.relu(var_2)
        var_4 = paddle.nn.functional.conv._conv_nd(var_3, self.parameter_3, bias=self.parameter_0, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_5 = paddle.nn.functional.activation.hardsigmoid(var_4, slope=0.2, offset=0.5)
        var_6 = paddle.tensor.math.multiply(x=var_0, y=var_5)
        return var_6



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 120, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 120, 68, 68], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 120, 68, 68]).astype('float32'),
    )
    return inputs


---------------------------------------gfl_gfl_r18vd_1x_coco-SIR_35---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[256, 256, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
           shape=[256, 128, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
           shape=[256, 512, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 128, 100, 152], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 256, 50, 76], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 512, 25, 38], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_13, bias=self.parameter_6, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_4 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_5, bias=self.parameter_10, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_5 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_15, bias=self.parameter_8, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = paddle.nn.functional.common.interpolate(var_5, scale_factor=2.0, mode='nearest')
        var_7 = var_4.__add__(var_6)
        var_8 = paddle.nn.functional.common.interpolate(var_7, scale_factor=2.0, mode='nearest')
        var_9 = var_3.__add__(var_8)
        var_10 = paddle.nn.functional.conv._conv_nd(var_9, self.parameter_2, bias=self.parameter_14, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_11 = paddle.nn.functional.conv._conv_nd(var_7, self.parameter_11, bias=self.parameter_3, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_12 = paddle.nn.functional.conv._conv_nd(var_5, self.parameter_4, bias=self.parameter_9, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_13 = paddle.nn.functional.conv._conv_nd(var_12, self.parameter_7, bias=self.parameter_1, stride=[2, 2], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_14 = paddle.nn.functional.activation.relu(var_13)
        var_15 = paddle.nn.functional.conv._conv_nd(var_14, self.parameter_12, bias=self.parameter_0, stride=[2, 2], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        return var_10, var_11, var_12, var_13, var_15



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 128, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 256, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 512, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 128, 100, 152], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 50, 76], dtype=paddle.float32),
        paddle.rand(shape=[1, 512, 25, 38], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 128, 100, 152]).astype('float32'),
        np.random.random(size=[1, 256, 50, 76]).astype('float32'),
        np.random.random(size=[1, 512, 25, 38]).astype('float32'),
    )
    return inputs


---------------------------------------MixNet_MixNet_S-SIR_43---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[120, 20, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[120, 20, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [22, 40, 28, 28], dtype: paddle.float32, stop_gradient: False)
    ):
        out = paddle.tensor.manipulation.split(var_0, [20, 20], axis=1)
        var_1 = out[0]
        var_2 = out[1]
        out = paddle.tensor.manipulation.split(var_0, [20, 20], axis=1)
        var_3 = out[0]
        var_4 = out[1]
        var_5 = paddle.nn.functional.conv._conv_nd(var_3, self.parameter_1, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = paddle.nn.functional.conv._conv_nd(var_4, self.parameter_0, bias=None, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = paddle.tensor.manipulation.concat((var_5, var_6,), axis=1)
        return var_7



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 40, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[22, 40, 28, 28], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[22, 40, 28, 28]).astype('float32'),
    )
    return inputs


---------------------------------------Twins_alt_gvt_base-SIR_157---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[96, 96],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[96, 192],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[192],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[96, 96, 8, 8],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [43, 3136, 96], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.common.linear(x=var_0, weight=self.parameter_0, bias=self.parameter_7, name=None)
        var_2 = var_1.reshape([43, 3136, 3, 32])
        var_3 = var_2.transpose([0, 2, 1, 3])
        var_4 = var_0.transpose([0, 2, 1])
        var_5 = var_4.reshape([43, 96, 56, 56])
        var_6 = paddle.nn.functional.conv._conv_nd(var_5, self.parameter_6, bias=self.parameter_2, stride=[8, 8], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = var_6.reshape([43, 96, 49])
        var_8 = var_7.transpose([0, 2, 1])
        var_9 = paddle.nn.functional.norm.layer_norm(var_8, normalized_shape=[96], weight=self.parameter_1, bias=self.parameter_4, epsilon=1e-05)
        var_10 = paddle.nn.functional.common.linear(x=var_9, weight=self.parameter_3, bias=self.parameter_5, name=None)
        var_11 = var_10.reshape([43, 49, 2, 3, 32])
        var_12 = var_11.transpose([2, 0, 3, 1, 4])
        var_13 = var_12.__getitem__(0)
        var_14 = var_12.__getitem__(1)
        var_15 = var_13.transpose([0, 1, 3, 2])
        return var_3, var_15, var_14



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[43, 3136, 96], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[43, 3136, 96]).astype('float32'),
    )
    return inputs


---------------------------------------CSWinTransformer_CSWinTransformer_base_384-SIR_598---------------------------------------
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [22, 32, 144, 24], dtype: paddle.float32, stop_gradient: True)
        var_1,    # (shape: [22, 32, 144, 24], dtype: paddle.float32, stop_gradient: True)
    ):
        var_2 = var_0.__add__(var_1)
        var_3 = var_2.transpose([0, 2, 1, 3])
        var_4 = var_3.reshape([-1, 144, 768])
        var_5 = var_0.shape[0]
        var_6 = var_4.reshape([var_5, 1, 1, 12, 12, 768])
        var_7 = var_6.transpose([0, 1, 3, 2, 4, 5])
        var_8 = var_7.reshape([var_5, 12, 12, 768])
        return var_8



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[22, 32, 144, 24], dtype=paddle.float32),
        paddle.rand(shape=[22, 32, 144, 24], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[22, 32, 144, 24]).astype('float32'),
        np.random.random(size=[22, 32, 144, 24]).astype('float32'),
    )
    return inputs


---------------------------------------CSWinTransformer_CSWinTransformer_base_384-SIR_283---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[96, 3, 7, 7],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [4, 3, 384, 384], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_3, bias=self.parameter_0, stride=[4, 4], padding=[2, 2], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_2 = var_1.flatten(start_axis=2, stop_axis=-1)
        var_3 = var_2.transpose([0, 2, 1])
        var_4 = paddle.nn.functional.norm.layer_norm(var_3, normalized_shape=[96], weight=self.parameter_1, bias=self.parameter_2, epsilon=1e-05)
        return var_4



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 3, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[4, 3, 384, 384], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[4, 3, 384, 384]).astype('float32'),
    )
    return inputs


---------------------------------------gfl_gfl_r50_fpn_1x_coco-SIR_66---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[1],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[68, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[80, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[68],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[80],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 256, 100, 152], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 256, 100, 152], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_2, bias=self.parameter_4, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_3 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_1, bias=self.parameter_3, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_4 = var_3.__mul__(self.parameter_0)
        return var_2, var_4



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 256, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 256, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 256, 100, 152], dtype=paddle.float32),
        paddle.rand(shape=[1, 256, 100, 152], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 256, 100, 152]).astype('float32'),
        np.random.random(size=[1, 256, 100, 152]).astype('float32'),
    )
    return inputs


---------------------------------------rotate_ppyoloe_r_ppyoloe_r_crn_l_3x_dota-SIR_179---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[192, 192, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[192],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 192, 128, 128], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 192, 1, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_0, bias=self.parameter_1, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_3 = paddle.tensor.ops.sigmoid(var_2)
        var_4 = var_0.__mul__(var_3)
        return var_4



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 192, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 192, 128, 128], dtype=paddle.float32),
        paddle.rand(shape=[1, 192, 1, 1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 192, 128, 128]).astype('float32'),
        np.random.random(size=[1, 192, 1, 1]).astype('float32'),
    )
    return inputs


---------------------------------------rotate_fcosr_fcosr_x50_3x_dota-SIR_92---------------------------------------
    def __init__(self):
        super().__init__()
    def forward(
        self,
        var_0,    # (shape: [1, 21824, 15], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 21824, 15], dtype: paddle.float32, stop_gradient: True)
        var_2,    # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = var_0.__sub__(var_1)
        var_4 = var_3.pow(2.0)
        var_5 = paddle.nn.functional.loss.binary_cross_entropy(var_0, var_1, weight=var_4, reduction='sum')
        var_6 = var_5.__truediv__(11)
        var_7 = var_6.__rmul__(1.0)
        var_8 = var_2.__rmul__(1.0)
        var_9 = var_7.__add__(var_8)
        return var_9, var_6



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1,), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 21824, 15], dtype=paddle.float32),
        paddle.rand(shape=[1, 21824, 15], dtype=paddle.float32),
        paddle.rand(shape=[1], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 21824, 15]).astype('float32'),
        np.random.random(size=[1, 21824, 15]).astype('float32'),
        np.random.random(size=[1]).astype('float32'),
    )
    return inputs


---------------------------------------GhostNet_GhostNet_x0_5-SIR_71---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[60],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[240, 60],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[240],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[60, 240],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [22, 240, 14, 14], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_0, output_size=1, data_format='NCHW', name=None)
        var_2 = paddle.tensor.manipulation.squeeze(var_1, axis=[2, 3])
        var_3 = paddle.nn.functional.common.linear(x=var_2, weight=self.parameter_1, bias=self.parameter_0, name=None)
        var_4 = paddle.nn.functional.activation.relu(var_3)
        var_5 = paddle.nn.functional.common.linear(x=var_4, weight=self.parameter_3, bias=self.parameter_2, name=None)
        var_6 = paddle.tensor.math.clip(x=var_5, min=0, max=1)
        var_7 = paddle.tensor.manipulation.unsqueeze(var_6, axis=[2, 3])
        var_8 = paddle.tensor.math.multiply(var_0, var_7)
        return var_8



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[22, 240, 14, 14], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[22, 240, 14, 14]).astype('float32'),
    )
    return inputs


---------------------------------------danet_danet_resnet50_os8_voc12aug_512x512_40k-SIR_31---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 512, 64, 64], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = var_0.shape
        var_2 = paddle.tensor.manipulation.reshape(var_0, (0, 512, -1,))
        var_3 = paddle.tensor.manipulation.reshape(var_0, (0, 512, -1,))
        var_4 = paddle.tensor.linalg.transpose(var_3, (0, 2, 1,))
        var_5 = paddle.tensor.linalg.bmm(var_2, var_4)
        var_6 = paddle.tensor.math.max(var_5, axis=-1, keepdim=True)
        var_7 = var_6.tile([1, 1, 512])
        var_8 = var_7.__sub__(var_5)
        var_9 = paddle.nn.functional.activation.softmax(var_8, axis=-1)
        var_10 = paddle.tensor.manipulation.reshape(var_0, (0, 512, -1,))
        var_11 = paddle.tensor.linalg.bmm(var_9, var_10)
        var_12 = var_1.__getitem__(2)
        var_13 = var_1.__getitem__(3)
        var_14 = paddle.tensor.manipulation.reshape(var_11, (0, 512, var_12, var_13,))
        var_15 = self.parameter_0.__mul__(var_14)
        var_16 = var_15.__add__(var_0)
        return var_16



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 512, 64, 64], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 512, 64, 64]).astype('float32'),
    )
    return inputs


---------------------------------------sparse_rcnn_sparse_rcnn_r50_fpn_3x_pro300_coco-SIR_33---------------------------------------
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[256, 256, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[256, 1024, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[256, 512, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
           shape=[256, 256, 3, 3],
           dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
           shape=[256, 2048, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 256, 144, 216], dtype: paddle.float32, stop_gradient: True)
        var_1,    # (shape: [1, 512, 72, 108], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 1024, 36, 54], dtype: paddle.float32, stop_gradient: False)
        var_3,    # (shape: [1, 2048, 18, 27], dtype: paddle.float32, stop_gradient: False)
    ):
        var_4 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_0, bias=self.parameter_12, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_5 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_3, bias=self.parameter_11, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = paddle.nn.functional.conv._conv_nd(var_2, self.parameter_1, bias=self.parameter_13, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_7 = paddle.nn.functional.conv._conv_nd(var_3, self.parameter_15, bias=self.parameter_14, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_8 = paddle.nn.functional.common.interpolate(var_7, scale_factor=2.0, mode='nearest')
        var_9 = var_6.__add__(var_8)
        var_10 = paddle.nn.functional.common.interpolate(var_9, scale_factor=2.0, mode='nearest')
        var_11 = var_5.__add__(var_10)
        var_12 = paddle.nn.functional.common.interpolate(var_11, scale_factor=2.0, mode='nearest')
        var_13 = var_4.__add__(var_12)
        var_14 = paddle.nn.functional.conv._conv_nd(var_13, self.parameter_9, bias=self.parameter_10, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_15 = paddle.nn.functional.conv._conv_nd(var_11, self.parameter_4, bias=self.parameter_2, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_16 = paddle.nn.functional.conv._conv_nd(var_9, self.parameter_5, bias=self.parameter_8, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_17 = paddle.nn.functional.conv._conv_nd(var_7, self.parameter_6, bias=self.parameter_7, stride=[1, 1], padding=[1, 1], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_18 = paddle.nn.functional.pooling.max_pool2d(var_17, 1, stride=2)
        return var_14, var_15, var_16, var_17, var_18



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 256, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 512, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 1024, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 2048, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec


def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 256, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 512, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 1024, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 2048, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 256, 144, 216], dtype=paddle.float32),
        paddle.rand(shape=[1, 512, 72, 108], dtype=paddle.float32),
        paddle.rand(shape=[1, 1024, 36, 54], dtype=paddle.float32),
        paddle.rand(shape=[1, 2048, 18, 27], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 256, 144, 216]).astype('float32'),
        np.random.random(size=[1, 512, 72, 108]).astype('float32'),
        np.random.random(size=[1, 1024, 36, 54]).astype('float32'),
        np.random.random(size=[1, 2048, 18, 27]).astype('float32'),
    )
    return inputs


