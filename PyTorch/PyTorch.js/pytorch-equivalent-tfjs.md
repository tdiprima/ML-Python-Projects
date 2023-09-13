## Is there a PyTorch Equivalent of tensorflow.js?

TL;DR &mdash; try ONNX.

<span style="color:#0f0;font-weight:bold;font-size:larger;">Do not mix these up.</span> <span style="font-size:27px;">🌱</span>

- **TorchJS**

    PyTorch offers a JavaScript version to use machine learning models in a browser or a JavaScript runtime, which is called [TorchJS](https://github.com/torch-js/torch-js#torchjs)

    TorchJS is a JS binding for PyTorch. Its primary objective is to allow running Torch Script inside Node.js program.

- **TorchScript**

    On the other hand, [TorchScript](https://pytorch.org/docs/stable/jit.html) is *not* a JavaScript implementation, but it is a way to serialize PyTorch models allowing them to be loaded and run in a non-Python environment (such as C++).

     TorchScript is not a JavaScript implementation of PyTorch.

- **[PyTorch.js](https://discuss.pytorch.org/t/is-there-a-pytorch-equivalent-of-tensorflow-js/143342)**

    [PyTorch.js](https://dev-discuss.pytorch.org/t/proposal-torch-js-a-javascript-frontend-for-pytorch/650) doesn't exist.

There are other libraries like [ONNX.js](https://hackernoon.com/how-to-run-machine-learning-models-in-the-browser-using-onnx), [TensorFlow.js](https://www.tensorflow.org/js) which support running machine learning models in a browser or a JavaScript runtime, in case you want to explore them.

For the most updated information, please check the latest PyTorch [documentation](https://pytorch.org/docs/stable/index.html) or their official [website](https://pytorch.org/tutorials/).

## TorchScript Tutorial

This tutorial is an introduction to TorchScript, an intermediate representation of a PyTorch model (subclass of `nn.Module`) that can then be run in a high-performance environment such as C++.

[Intro to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)

## Train PyTorch in JS

Convert a PyTorch model and train it in JavaScript in your browser using ONNX Runtime Web: [train-pytorch-in-js](https://github.com/juharris/train-pytorch-in-js)

<br>
