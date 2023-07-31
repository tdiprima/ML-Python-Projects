## Convolutional Neural Network (CNN)

<span style="color:#0000dd;">Why would we use it?  What is it known for?  Why is it useful?</span>

A CNN "looks" at pictures or videos and figures out what's in them.

It does this by breaking the image up into tiny pieces ("pixels") and analyzing each piece to see if it matches any patterns that it has learned from other pictures.

<!-- https://data-flair.training/blogs/cats-dogs-classification-deep-learning-project-beginners/ -->

<!--<img src="https://data-flair.training/blogs/wp-content/uploads/sites/2/2020/05/Cats-Dogs-Classification-deep-learning.gif" width="650">-->

One of the main things that a CNN is known for is its ability to **recognize objects in pictures**.

For example, if you show a CNN a bunch of pictures of cats and a bunch of pictures of dogs, it can learn to tell the difference between them.

This can be really helpful in all sorts of situations, from identifying animals in wildlife photos to detecting tumors in medical images.

Another thing that makes CNNs really powerful is their ability to **"generalize"** what they've learned.

This means that if you show a CNN a new picture that it hasn't seen before, it can still use its knowledge of patterns to make an educated guess about what's in the picture. <span style="color:#ff00cc;">Like figuring out people in your photo album.</span>

Like, if a CNN has seen lots of pictures of cars, it can probably recognize a new picture of a car even if it's a different color or from a different angle. <span style="color:deeppink;">The Carvana challenge.</span>

<!--
Image classification
<img src="https://149695847.v2.pressablecdn.com/wp-content/uploads/2017/09/localizationVsDetection.png" width="600">
-->
<!--<i style="all:revert">What</i>-->

<!-- https://www.tutorialrepublic.com/faq/how-to-reset-or-remove-css-style-for-a-particular-element.php -->

## Pro-tip!

When you get:

```
"ModuleNotFoundError: No module named 'keras.models'; 'keras' is not a package"
```

<br>
<span style="color:red;font-size:larger">"Dr. Snoopy" says:</span>

The problem is that you have a file named "keras.py" and this shadows the real keras package.

Don't do that, never name a python script the same as a package.

Solution is to rename your script.

I knew that. 🤦‍♀️

<br>
