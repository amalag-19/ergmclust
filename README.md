# netclust
The netclust package is an R implementation to cluster and estimate parameters in Exponential Family Random Graph Models popularly known as ERGMs.

For static undirected and directed networks, the models were originally proposed in Vu et. al., 2013. In my Ph.D. I extended the research by developing an elaborate framework that clusters network data usually encountered in real life situations, such as social networks where communities evolve with time, product review networks where customers review different products on an ecommerce platform in a bipartite setting, [weighted river networks](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mVExpwIAAAAJ&citation_for_view=mVExpwIAAAAJ:LkGwnXOMwfcC) with applications to safeguard water quality standards, trade networks, email networks, etc.

To scale up the clustering inference for large networks, the estimation procedure is based on novel stochastic Variational Expectation Maximization algorithms. I coded these algorithms in `R` and used `C++` to speed up several time consuming routines for parameter updates.

A brute force approach requires O($K^N$) time complexity where K is the number of clusters and N is the number of nodes. The variational expectation maximization approach works by constructing a tight lower bound to the log likelihood thereby significantly boosting the time complexity to O($N^2K^2$).

Further taking advantage of stochastic estimation, O($N^2K^2$) could be reduced to O($N^2$) by sub-sampling the cluster memberships of nodes.
