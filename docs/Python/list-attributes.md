## List attributes of dataset

<span style="color:#0000dd;">I set up my training data; it's of type torchvision.datasets.mnist.FashionMNIST.</span>

<span style="color:#0000dd;">It has a .classes attribute.  I wanted to see what the other attributes were.  So I did vars(train_data).  I don't even see ".classes" in the list.  Why?</span>

The **`vars()`** function returns a dictionary containing the attributes and their values of an object. However, some attributes may not be included in the dictionary if they are implemented using properties or other methods that are not stored as regular instance variables.

In the case of `torchvision.datasets.mnist.FashionMNIST`, the `.classes` attribute is **implemented as a property**, which means that it is **calculated dynamically** when accessed and is not actually stored as an instance variable. Therefore, it will not appear in the dictionary returned by `vars(train_data)`.

Instead of using `vars()`, you can use the **`dir()`** function to **list all the attributes** of an object, including **properties:**

```python
print(dir(train_data))
# OR
import pprint as pp
pp.pprint(dir(train_data))
```

This will give you a list of all the **attributes** and **methods** of `train_data`, including the `classes` property. You can then access it using `train_data.classes`.
