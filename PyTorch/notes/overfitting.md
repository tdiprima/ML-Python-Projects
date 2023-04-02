## Early Stopping with PyTorch to Restrain your Model from Overfitting

https://medium.com/analytics-vidhya/early-stopping-with-pytorch-to-restrain-your-model-from-overfitting-dce6de4081c5

Also: https://github.com/Bjarten/early-stopping-pytorch

And again: https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting

A lot of machine learning algorithm developers, especially the newcomer worries about how much epochs should I select for my model training. Hopefully, this article will help you to find a solution to that confusion.

Prior to beginning, let’s know about what is epoch, overfitting, and what is early stopping and why we will use that?

Well, very simply, EPOC is the number of the forward pass and back-propagation will happen in your network with your whole training data. This means how many times your model will get to know about your data before test it.

To say about overfitting, I am going back to high school Chemistry, Probably you remember the critical point of chemical reaction! Right? At your chemistry lab do you remember about the titration? There was a critical point for the amount of Acid and Base to make a perfect titration because if you had dropped a couple of extra droplets of the base in acid solution, your titration did not use to show the perfect result. Which means at some point you have to stop which will be balanced for both. The ideology of Overfitting is similar. You have to stop training your model at some stage. If you don’t do so your model will be biased training data like imbalance situation in titration. So, early stopping is that stage where you have to stop that training your model.

So what do we need to do for early stopping? We can push a validation set of data to continuously observe our model whether it’s overfitting or not. Also you can see a well discussed article on Hackernoon on overfitting.


Early Termination Point [1]
As you can see, the errors were more or less similar since the beginning. However, at some point, the difference is increasing, which indicates we need to stop the training early with respect to error and Epochs.

Most of the Machine Learning libraries come with early stopping facilities. For example, Keras Early Stopping is Embedded with the Library. You can see over here, it’s a fantastic article on that.

On top of my head, I know PyTorch’s early stopping is not Embedded with the library. However, it’s official website suggests another library that fits with it and can have an eye on the Model at the training stage. It’s Ignite, you will find more implementation documentation over there.

However, I am using another third-party library which was cool according to my usability. It’s in GitHub, the authors named it as TorchSample. You can just simply download from there, it’s open-source, and you can customize it if you need to do so.

Now, let’s start how to work with that, Let’s have a very simple LSTM model

```py
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.embededLayer = nn.Embedding(num_embeddings =MAX_LENGTH, embedding_dim = VOCAB_SIZE, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
        self.lstmCells = nn.LSTM(VOCAB_SIZE, HIDDEN_IN, MAX_LENGTH)   #nn.LSTM(input_size, hidden_size, num_layers) 
        self.linearLayer = nn.Linear(128, 32)  # equivalent to Dense in keras
        self.dropOut = nn.Dropout(0.2)
        self.linearLayer2 = nn.Linear(32, 1) 
        self.reluAct = nn.ReLU()
        self.softAct = nn.Softmax()
        self.logSoftAct = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        clsf = self.embededLayer(x)
        clsf, _ = self.lstmCells(clsf)
        clsf = self.linearLayer(clsf[:,-1,:])
        clsf = self.reluAct(clsf)
        clsf = self.linearLayer2(clsf)
        clsf = self.sigmoid(clsf)
        return clsf
```

Generally, we follow this strategy and follow up the experiment, such that, here I put number of epochs is 200 based on my hypothesis for training.

```py
model = Network()
model.compile(loss='nll_loss', optimizer='adam')
model.fit(x_train, y_train, val_data=(x_test, y_test),num_epoch=200, batch_size=128, verbose=1)
loss = model.evaluate(x_train, y_train)
y_pred = model.predict(x_train)
```

However, if my Model converge after 20 epochs only then we have to stop training right away. So how can we do that? Simply, just import the package and write a small portion of code by yourself.

Now you have to import The ModuleTrainer class, which provides a high-level training interface which abstracts away the training loop while providing callbacks, constraints, initializers, regularizers, and more.

```py
from torchsample.modules import ModuleTrainer
trainer = ModuleTrainer(model)
model = ModuleTrainer(Network())
model.compile(loss='nll_loss', optimizer='adam')
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
model.set_callbacks(callbacks)
model.fit(x_train, y_train, val_data=(x_test, y_test),num_epoch=200, batch_size=128, verbose=1)
loss = model.evaluate(x_train, y_train)
y_pred = model.predict(x_train)
```

Here you can see, I just passed the model through the model trainer and creating a callback function to keep track of validation. Based on my validation error I will halt the training process. There is another parameter name patience which allows how many epochs it will wait for to terminate if models start overfitting. For example, if the model starts showing the variation than the previous loss at 31st epochs it will wait until the next 5 epochs and if still, the loss doesn’t improve then it will halt the training and return the model as done with the training and that’s all.

Hope you have enjoyed the article. On my next article I will try to write about Bayesian Hyperparameters Optimization and Regularization.

* Deep Learning
* Early Stopping
* Pytorch
* Pytorch Early Stopping

<br>
