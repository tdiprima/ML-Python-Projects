### 𝑓𝑦(𝑥)

<span style="color:#a71f36">Some people were discussing logits in deep learning, saying "Logit is referred to the output of a function just before its normalization. So if for label 𝑦 we have score 𝑓𝑦(𝑥)..."</span>

## Logits in Deep Learning 🍋

Sure, I'd be happy to help clarify what 𝑓𝑦(𝑥) means! 

Think about this: you and your friends have a lemonade stand. You want to know which day of the week is the best to sell lemonade. To do this, you keep track of the weather, the temperature, how many people walk by, and other things that might affect how much lemonade you sell. Each of these things is like an "x" &ndash; they're pieces of information you're using to try and predict something.

Now, let's say you have a calculator that can take all this information (or data) and give you a score for each day of the week, predicting how good it will be for selling lemonade. The function that calculator uses to do this is 𝑓, and when you plug in your data (x) for each day, you get out a score. 

The label 𝑦 is what you're trying to predict. In this case, that's the best day to sell lemonade. So, 𝑓𝑦(𝑥) is just the score that your magic calculator gives for that day, based on the data you plugged in.

In a deep learning model, we often use these scores, or logits, as part of the process to make our final prediction. We usually pass them through another function, called a "softmax function", that turns the scores into probabilities for each label. So, in the end, we get a list of probabilities for each day, telling us how likely it is to be the best day for selling lemonade.

Remember, the logits (scores) are the raw outputs from our magic calculator (deep learning model) for each label before they get normalized, or adjusted, into probabilities.



### English

*fy(x)* would be pronounced "eff sub why of ex".

It means that given some input value *x*, the function *f* generates a set of output scores for each possible label, one of which is *y*.

So *fy(x)* represents the score or logit assigned by the function *f* to the label *y* for the input *x*.

<span style="color:#997fff;">It's like the function signature; denoting the function, its arguments, and return variable.</span>

These logits can then be normalized using the softmax function to obtain a probability distribution over the possible labels.

In other words, the softmax function takes the logits as input and maps them to a set of probabilities that add up to one, representing the model's estimate of the likelihood of each label given the input *x*.

### See: [What does the logit value actually mean?](https://stats.stackexchange.com/questions/52825/what-does-the-logit-value-actually-mean#498922)

Logit: "The (unnormalized) score for a label or (functional confidence) for a specific class/label."

Logit is referred to **the output of a function** (e.g. a Neural Net) **just before its normalization** (which we usually use the softmax).

<br>
