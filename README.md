# ergmclust
The ergmclust package is an R implementation to cluster and estimate parameters in Exponential Family Random Graph Models popularly known as ERGMs.

For static undirected and directed networks, the models were originally proposed in Vu et. al., 2013. In my Ph.D. I extended the research by developing an elaborate framework that clusters network data usually encountered in real life situations, such as social networks where communities evolve with time, product review networks where customers review different products on an ecommerce platform in a bipartite setting, [weighted river networks](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mVExpwIAAAAJ&citation_for_view=mVExpwIAAAAJ:LkGwnXOMwfcC) with applications to safeguard water quality standards, trade networks, email networks, etc.

This readme is a guided tour to a basic introduction to these models in laymen terms (for technical details, you can always read my peer reviewed technical articles), their applicability in day to day real world applications and future scope.


## Literature Overview

Here is a brief timeline of the research in the field of Exponential Family Random Graph Models –

<img width="871" alt="Screenshot 2024-02-06 at 11 02 40 PM" src="https://github.com/amalag-19/netclust/assets/10363788/845606b7-5f0d-4bdb-825e-8fab4f77fe3c">

## Algorithm
Estimation algorithms in ERGMs are based on variational expectation maximization. That's a bunch of jargon, that I break down below

### Expectation-Maximization (EM) Algorithm:
Imagine you're trying to figure out the heights of people in a dark room, but you can only measure their shadows. In EM, you have two steps:

1. Expectation Step (E-step): Make an educated guess about their heights based on the shadows.
2. Maximization Step (M-step): Adjust your guess about their heights based on the shadows and any other prior knowledge you have.

### Variational EM: 
Now, let's add some flexibility. Instead of being absolutely certain about your guess (like in traditional EM), you allow for some uncertainty. It's like saying, "I'm pretty sure this person is around 6 feet tall, but there's a chance they could be a bit taller or shorter." This flexibility lets you capture more complex patterns in your data.

### How it Works:
Variational Step: Introduce a set of parameters that represent this uncertainty. For example, instead of just guessing one height for each person, you guess a range of possible heights.
Expectation-Maximization Steps: Alternate between refining your guesses about the heights and updating the parameters that represent the uncertainty. In each iteration, try to find the best combination of heights and uncertainties that explain the shadows you observe.

### Why it's Useful:
Variational EM is handy when your data is complex and doesn't fit neatly into simple patterns. By allowing for uncertainty in your guesses, you can capture more nuances in the data and get better insights.

In essence, Variational EM is like trying to solve a puzzle in a dimly lit room where you're not entirely sure about the shapes of the pieces. You make educated guesses, adjust them based on what you see, and keep refining your understanding until you find the best fit, all while acknowledging that there might be some ambiguity in your observations.

A picture is worth a thousand words, so I'll draw a less technical figure to illustrate the workings of Variational EM algorithm and how the lower bound (ELBO) is maximized iteratively.
<img width="917" alt="image" src="https://github.com/amalag-19/ergmclust/assets/10363788/5f088bc2-899f-4302-9529-bcdb2d977bc9">


## Implementation


## Scalability

To scale up the clustering inference for large networks, the estimation procedure is based on novel stochastic Variational Expectation Maximization algorithms. I coded these algorithms in `R` and used `C++` to speed up several time consuming routines for parameter updates.

A brute force approach requires O($K^N$) time complexity where K is the number of clusters and N is the number of nodes. The variational expectation maximization approach works by constructing a tight lower bound to the log likelihood thereby significantly boosting the time complexity to O($N^2K^2$).

Further taking advantage of stochastic estimation, O($N^2K^2$) could be reduced to O($N^2$) by sub-sampling the cluster memberships of nodes as part of the soft clustering during the algorithm.





