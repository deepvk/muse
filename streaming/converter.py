import argparse
import gdown
import importlib
import numbers
import shutil
from typing import Tuple
from pathlib import Path

import torch
import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer


@nobuco.converter(
    torch.nn.functional.glu,
    channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS,
)
def torch_glu(input, dim=1):
    def tf_glu(input, dim=1):
        ax = -1 if dim == 1 else dim
        out, gate = tf.split(input, 2, axis=ax)
        gate = tf.sigmoid(gate)
        return tf.multiply(out, gate)

    return lambda input, dim=1: tf_glu(input, dim)


@nobuco.converter(torch.Tensor.all)
def tensor_all(input: torch.Tensor):
    return lambda input: tf.math.reduce_all(input)


@nobuco.converter(torch.atan2)
def atan2(input_x: torch.Tensor, input_y: torch.Tensor):
    return lambda input_x, input_y: tf.math.atan2(input_x, input_y)


@nobuco.converter(torch.Tensor.std)
def tensor_std(input: torch.Tensor, dim, keepdim):
    return lambda input, dim, keepdim: tf.math.reduce_std(
        input, axis=dim, keepdims=keepdim
    )


@nobuco.converter(torch.Tensor.t)
def tensor_t(input: torch.Tensor):
    return lambda input: tf.transpose(input)


@nobuco.converter(torch.Tensor.__getattribute__)
def torch_getattribute_complex_resolve(input: torch.Tensor, attr: str):
    def tf_complex_getattribute(input, attr):
        if attr == "real":
            tf_func = tf.math.real
        elif attr == "imag":
            tf_func = tf.math.imag
        else:
            tf_func = lambda x: getattr(x, attr)
        return tf_func(input)

    return tf_complex_getattribute


@nobuco.converter(torch.nn.Conv2d)
def converter_Conv2d(self, input: torch.Tensor):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation

    out_filters, in_filters, kh, kw = weight.shape

    weights = weight.cpu().detach().numpy()
    weights = tf.transpose(weights, (2, 3, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if padding != 0 and padding != (0, 0) and padding != "valid" and padding != "same":
        pad_layer = tf.keras.layers.ZeroPadding2D(padding)
    else:
        pad_layer = None

    pad_arg = padding if padding == "same" else "valid"
    conv = tf.keras.layers.Conv2D(
        filters=out_filters,
        kernel_size=(kh, kw),
        strides=stride,
        padding=pad_arg,
        dilation_rate=dilation,
        groups=groups,
        use_bias=use_bias,
        weights=params,
    )

    def func(input):
        if pad_layer is not None:
            input = pad_layer(input)
        output = conv(input)
        return output

    return func


@nobuco.converter(torch.nn.Conv1d)
def converter_Conv1d(self, input: torch.Tensor):
    weight = self.weight
    bias = self.bias
    groups = self.groups
    padding = self.padding
    stride = self.stride
    dilation = self.dilation

    out_filters, in_filters, kw = weight.shape
    weights = weight.cpu().detach().numpy()
    weights = tf.transpose(weights, (2, 1, 0))

    if bias is not None:
        biases = bias.cpu().detach().numpy()
        params = [weights, biases]
        use_bias = True
    else:
        params = [weights]
        use_bias = False

    if isinstance(padding, numbers.Number):
        padding = (padding,)
    if padding != (0,) and padding != "valid" and padding != "same":
        pad_layer = tf.keras.layers.ZeroPadding1D(padding[0])
    else:
        pad_layer = None

    pad_arg = padding if padding == "same" else "valid"
    conv = tf.keras.layers.Conv1D(
        filters=out_filters,
        kernel_size=kw,
        strides=stride,
        padding=pad_arg,
        dilation_rate=dilation,
        groups=groups,
        use_bias=use_bias,
        weights=params,
    )

    def func(input):
        if pad_layer is not None:
            input = pad_layer(input)
        output = conv(input)
        return output

    return func


@nobuco.converter(
    torch.concat,
    channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS,
)
def converter_concat(tensors: Tuple[torch.Tensor], dim):
    def tf_concat(tensors, dim):
        return tf.concat(list(tensors), -dim)

    return tf_concat


def main(args, config):
    try:
        Path(config.original_model_dst).mkdir(exist_ok=False)
    except FileExistsError:
        shutil.rmtree(config.original_model_dst)
    shutil.copytree(config.original_model_src, config.original_model_dst)
    py_module = importlib.import_module(args.model_py_module)
    cls_model = getattr(py_module, args.class_name)
    model = cls_model(
        source=["drums", "bass", "other", "vocals"],
        depth=4,
        channel=28,
        bottlneck_lstm=False,
        stft_flag=False,
    )

    if model.bottlneck_lstm:
        weights_path = config.weights_dir / "weight_LSTM.pt"
        gdrive_id = config.gdrive_weights_LSTM_id
    else:
        weights_path = config.weights_dir / "weight_conv.pt"
        gdrive_id = config.gdrive_weights_conv_id
    try:
        config.weights_dir.mkdir(parents=True, exist_ok=False)
        download_weights = True
    except FileExistsError:
        try:
            Path(weights_path).touch(exist_ok=False)
            download_weights = True
        except FileExistsError:
            download_weights = False
    if download_weights:
        gdown.download(id=gdrive_id, output=str(weights_path))

    model.load_state_dict(
        torch.load(str(weights_path), map_location=torch.device("cpu"))
    )

    model = model.eval()

    class OuterSTFT:
        def __init__(self, length_wave, model):
            self.length_wave = length_wave
            self.model = model

        def stft(self, wave):
            return self.model.stft.stft(wave)

        def istft(self, z):
            return self.model.stft.istft(z, self.length_wave)

    SEGMENT_WAVE = 44100
    dummy_wave = torch.rand(size=(1, 2, SEGMENT_WAVE))
    dummy_spectr = OuterSTFT(SEGMENT_WAVE, model).stft(dummy_wave)

    keras_model = nobuco.pytorch_to_keras(
        model,
        args=[dummy_spectr],
        kwargs=None,
        inputs_channel_order=ChannelOrder.PYTORCH,
    )

    model_path = str(
        args.out_dir + f"/{args.class_name}_outer_stft_{SEGMENT_WAVE / 44100:.1f}"
    )

    keras_model.save(model_path + ".h5")
    custom_objects = {"WeightLayer": WeightLayer}

    converter = TFLiteConverter.from_keras_model_file(
        model_path + ".h5", custom_objects=custom_objects
    )
    converter.target_ops = [
        tf.lite.OpsSet.SELECT_TF_OPS,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    tflite_model = converter.convert()

    with open(model_path + ".tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converter script")
    from config.config import ConverterConfig

    config = ConverterConfig()

    parser.add_argument(
        "-I",
        dest="model_py_module",
        help="py module of model\nformat: pkg.mod e.g model.PM_Unet",
        default=config.model_py_module,
        type=str,
    )
    parser.add_argument(
        "-C",
        dest="class_name",
        help="class name of nn.Module",
        default=config.model_class_name,
        type=str,
    )
    parser.add_argument(
        "-O",
        dest="out_dir",
        help="specified output dir",
        default=config.tflite_model_dst,
        type=str,
    )

    args = parser.parse_args()
    main(args, config)
