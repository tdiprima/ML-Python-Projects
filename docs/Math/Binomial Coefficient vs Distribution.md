## Binomial Coefficient Explained 🏀

Let's start with a fun scenario. Imagine you and four of your friends have formed a basketball team, and you're all so good that any one of you could be the team captain. But only one person can be the captain at a time.

How many different ways can you choose a captain for your team? Well, you can choose from 5 people, so there are 5 ways to do it.

But what if you wanted to choose two captains? That's where the binomial coefficient comes in!

You might think that because there are 5 ways to choose the first captain and 4 ways to choose the second captain, there would be 20 ways to choose two captains.

However, you'd be over-counting because the order in which you pick the captains doesn't matter - picking you first and then your friend is the same as picking your friend first and then you. 

So, we need to adjust our calculation to remove these duplicates. The binomial coefficient does exactly this. It tells us how many different ways we can choose a certain number of items (like basketball captains) from a larger set (like your whole team), without worrying about the order in which we choose them.

The binomial coefficient is often represented like this: (5 choose 2), and the formula for it is:

```
n!
----
r!(n-r)!
```

<br>
where "n" is the total number of items (in this case, 5 team members), "r" is the number of items you're choosing (in this case, 2 captains), and the "!" symbol represents a factorial (which means you multiply all the numbers from that number down to 1).

So, using this formula, the number of ways to choose 2 captains from 5 team members is:

```
5! = 5*4*3*2*1 = 120
2! = 2*1 = 2
(5-2)! = 3*2*1 = 6
```

<br>
Plug these into the formula and you get:

```c
120 / (2 * 6) = 10
```

<br>
So, there are 10 different ways you can choose 2 captains from your 5 member basketball team.

## Binomial Distribution 🪙

Let's think about flipping a coin. You know that when you flip a coin, you have a 50% chance of it landing on heads, and a 50% chance of it landing on tails. That's because there are only two possible outcomes.

Now, imagine you decided to flip the coin 10 times in a row. You could ask yourself a question like, "What's the chance that I will get exactly 6 heads when I flip this coin 10 times?" This is where the binomial distribution comes in.

The binomial distribution is a way of figuring out the probability of getting a certain number of "successes" (like flipping heads) in a set number of independent trials (like 10 coin flips), given the probability of success on each trial (50% chance to flip heads).

Here are the important parts you need to know for a binomial distribution:

1. Number of trials (n): This is how many times you're doing something, like flipping a coin 10 times.

2. Probability of success (p): This is the chance of getting the outcome you're interested in each time you do the trial, like a 50% chance of flipping heads each time.

3. Number of successes (k): This is the outcome you're trying to find the probability of, like flipping exactly 6 heads.

The formula to calculate a binomial distribution looks a little complicated, but I'll explain it:

```c
P(k; n, p) = C(n, k) * (p^k) * ((1-p)^(n-k))
```

<br>
Here, 

- P(k; n, p) is the probability of getting k successes in n trials.

- C(n, k) is the binomial coefficient we just talked about (which is how many different ways you can get k successes in n trials).

- p^k is the probability of getting a success k times.

- (1-p)^(n-k) is the probability of not getting a success for the rest of the trials.

So, to find the chance of getting exactly 6 heads in 10 coin flips, you would plug in those numbers to find the answer.

It's important to remember that the binomial distribution assumes each trial is independent, which means the result of one trial (like one coin flip) doesn't affect the results of the other trials. This is true for coin flips because getting heads on one flip doesn't make you more or less likely to get heads on the next flip.

So that's the binomial distribution! It's a way of calculating the chances of getting a certain number of successes in a certain number of trials. It's like a more complex version of flipping a coin and seeing whether it comes up heads or tails.

## Hat ^

The `^` symbol as used in `(p^k)` actually represents exponentiation. 

So, `p^k` means "p raised to the power of k". In other words, you're multiplying p by itself k times.

For example, if p is 0.5 (like the probability of getting heads in a coin toss), and k is 2 (meaning you want to find the probability of getting heads twice in a row), then `p^k` would be `(0.5)^2`, which equals 0.25. 

In general, the term `(p^k)` in the binomial distribution formula represents the probability of getting k successes (like flipping heads k times).


## math.comb() and comb()

You can use the `math` library to calculate binomial coefficients using the `comb()` function...

```python
import math

# Calculate binomial coefficient of 5 choose 2
n = 5
k = 2
coef = math.comb(n, k)

print(coef)  # Output: 10
```

<br>

Note that the `math.comb()` function was added in Python 3.8. If you are using an older version of Python, you can use the `scipy.special.comb()` function from the `scipy` library instead...

```python
from scipy.special import comb

# Calculate binomial coefficient of 5 choose 2
n = 5
k = 2
coef = comb(n, k)

print(coef)  # Output: 10.0
```

<br>
