## Dir() vs Vars() in Python

Sure! Imagine you have a toy robot that you can program. Let's call the robot "Robo."

1. `dir(Robo)` is like asking, "Hey Robo, what all tricks can you do?" Robo might say, "I can walk, talk, dance, and even sing!" It shows you all the functions (methods) and properties (attributes) that Robo has.
   
2. `vars(Robo)` is more like asking, "Hey Robo, what's in your backpack?" And Robo might show you a small screen that says, "I have a battery level of 90%, my current speed is 2 mph, and my color is blue." It shows you all the variables that Robo is currently keeping track of.

Both `dir()` and `vars()` are used to inspect what an object (like our Robo) contains, but they do it in different ways.

### What happens when things are calculated dynamically?

1. **dir()**: Even if Robo learns a new trick, like flipping, `dir(Robo)` will immediately tell you, "Hey, now I can also flip!" It'll list all the tricks Robo can do, whether they were there from the beginning or added later.

2. **vars()**: If Robo finds a new item, like a "laser pointer," and puts it in his backpack, `vars(Robo)` will show that too. It will say, "Now, I also have a laser pointer in my backpack!"

    * However! Some attributes may not be included in the dictionary if they are implemented using properties or other methods that are not stored as regular instance variables.

    * In the case of `torchvision.datasets.mnist.FashionMNIST`, the `.classes` attribute is implemented as a property, which means that it is calculated dynamically when accessed and is not actually stored as an instance variable. Therefore, it will not appear in the dictionary returned by `vars(train_data)`.

So, in simple terms:

`dir()` shows you all the capabilities (methods and attributes) of an object, whether they are set from the beginning or calculated (added) later.

`vars()` shows you the current state of an object, like what variables it's tracking at the moment.


## List attributes of dataset

Anyway, so instead of using `vars()`, you can use the **`dir()`** function to list all the attributes of an object, including properties:

```python
print(dir(train_data))

import pprint as pp
pp.pprint(dir(train_data))
```

<br>

This will give you a list of all the attributes and methods of `train_data`, including the `classes` property. You can then access it using `train_data.classes`.

<br>

