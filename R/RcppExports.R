# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

rowsum_dir_Mat <- function(M) {
    .Call('_ergmclust_rowsum_dir_Mat', PACKAGE = 'ergmclust', M)
}

colsum_dir_Mat <- function(M) {
    .Call('_ergmclust_colsum_dir_Mat', PACKAGE = 'ergmclust', M)
}

gamma_update_HMM_stat_dir <- function(gamma, pi, theta, network, N, K) {
    .Call('_ergmclust_gamma_update_HMM_stat_dir', PACKAGE = 'ergmclust', gamma, pi, theta, network, N, K)
}

grad_HMM_stat_dir_oe <- function(theta, gamma, network, N, K) {
    .Call('_ergmclust_grad_HMM_stat_dir_oe', PACKAGE = 'ergmclust', theta, gamma, network, N, K)
}

grad_HMM_stat_dir_re <- function(theta, gamma, network, N, K) {
    .Call('_ergmclust_grad_HMM_stat_dir_re', PACKAGE = 'ergmclust', theta, gamma, network, N, K)
}

hess_HMM_stat_dir_oe <- function(theta, gamma, N, K) {
    .Call('_ergmclust_hess_HMM_stat_dir_oe', PACKAGE = 'ergmclust', theta, gamma, N, K)
}

hess_HMM_stat_dir_re <- function(theta, gamma, N, K) {
    .Call('_ergmclust_hess_HMM_stat_dir_re', PACKAGE = 'ergmclust', theta, gamma, N, K)
}

hess_HMM_stat_dir_oe_re <- function(theta, gamma, N, K) {
    .Call('_ergmclust_hess_HMM_stat_dir_oe_re', PACKAGE = 'ergmclust', theta, gamma, N, K)
}

ELBO_conv_HMM_stat_dir <- function(gamma, alpha, theta, network, N, K) {
    .Call('_ergmclust_ELBO_conv_HMM_stat_dir', PACKAGE = 'ergmclust', gamma, alpha, theta, network, N, K)
}

grad_HMM_stat_dir_oe_K1 <- function(theta, network, N) {
    .Call('_ergmclust_grad_HMM_stat_dir_oe_K1', PACKAGE = 'ergmclust', theta, network, N)
}

grad_HMM_stat_dir_re_K1 <- function(theta, network, N) {
    .Call('_ergmclust_grad_HMM_stat_dir_re_K1', PACKAGE = 'ergmclust', theta, network, N)
}

hess_HMM_stat_dir_oe_K1 <- function(theta, N) {
    .Call('_ergmclust_hess_HMM_stat_dir_oe_K1', PACKAGE = 'ergmclust', theta, N)
}

hess_HMM_stat_dir_re_K1 <- function(theta, N) {
    .Call('_ergmclust_hess_HMM_stat_dir_re_K1', PACKAGE = 'ergmclust', theta, N)
}

hess_HMM_stat_dir_oe_re_K1 <- function(theta, N) {
    .Call('_ergmclust_hess_HMM_stat_dir_oe_re_K1', PACKAGE = 'ergmclust', theta, N)
}

ELBO_conv_HMM_stat_dir_K1 <- function(theta, network, N) {
    .Call('_ergmclust_ELBO_conv_HMM_stat_dir_K1', PACKAGE = 'ergmclust', theta, network, N)
}

rowsum_Mat_new <- function(M) {
    .Call('_ergmclust_rowsum_Mat_new', PACKAGE = 'ergmclust', M)
}

colsum_Mat_new <- function(M) {
    .Call('_ergmclust_colsum_Mat_new', PACKAGE = 'ergmclust', M)
}

epan <- function(input) {
    .Call('_ergmclust_epan', PACKAGE = 'ergmclust', input)
}

gamma_update_weighted_stat_undir <- function(gamma, pi, theta, block_dens_mat, adjmat, N, K) {
    .Call('_ergmclust_gamma_update_weighted_stat_undir', PACKAGE = 'ergmclust', gamma, pi, theta, block_dens_mat, adjmat, N, K)
}

grad_theta_weighted_stat_undir <- function(theta, gamma, adjmat, N, K) {
    .Call('_ergmclust_grad_theta_weighted_stat_undir', PACKAGE = 'ergmclust', theta, gamma, adjmat, N, K)
}

hess_theta_weighted_stat_undir <- function(theta, gamma, N, K) {
    .Call('_ergmclust_hess_theta_weighted_stat_undir', PACKAGE = 'ergmclust', theta, gamma, N, K)
}

tie_clust_partition <- function(clust_est, adjmat, wtmat, N, K) {
    .Call('_ergmclust_tie_clust_partition', PACKAGE = 'ergmclust', clust_est, adjmat, wtmat, N, K)
}

ELBO_conv_weighted_stat_undir <- function(gamma, pi, theta, block_dens_mat, adjmat, N, K) {
    .Call('_ergmclust_ELBO_conv_weighted_stat_undir', PACKAGE = 'ergmclust', gamma, pi, theta, block_dens_mat, adjmat, N, K)
}

grad_theta_weighted_stat_undir_K1 <- function(theta, adjmat, N) {
    .Call('_ergmclust_grad_theta_weighted_stat_undir_K1', PACKAGE = 'ergmclust', theta, adjmat, N)
}

hess_theta_weighted_stat_undir_K1 <- function(theta, N) {
    .Call('_ergmclust_hess_theta_weighted_stat_undir_K1', PACKAGE = 'ergmclust', theta, N)
}

ELBO_conv_weighted_stat_undir_K1 <- function(theta, block_dens_mat, adjmat, N) {
    .Call('_ergmclust_ELBO_conv_weighted_stat_undir_K1', PACKAGE = 'ergmclust', theta, block_dens_mat, adjmat, N)
}

rowsum_Mat <- function(M) {
    .Call('_ergmclust_rowsum_Mat', PACKAGE = 'ergmclust', M)
}

colsum_Mat <- function(M) {
    .Call('_ergmclust_colsum_Mat', PACKAGE = 'ergmclust', M)
}

gamma_update_HMM_stat_undir <- function(gamma, pi, theta, network, N, K) {
    .Call('_ergmclust_gamma_update_HMM_stat_undir', PACKAGE = 'ergmclust', gamma, pi, theta, network, N, K)
}

grad_HMM_stat_undir <- function(theta, gamma, network, N, K) {
    .Call('_ergmclust_grad_HMM_stat_undir', PACKAGE = 'ergmclust', theta, gamma, network, N, K)
}

hess_HMM_stat_undir <- function(theta, gamma, N, K) {
    .Call('_ergmclust_hess_HMM_stat_undir', PACKAGE = 'ergmclust', theta, gamma, N, K)
}

ELBO_conv_HMM_stat_undir <- function(gamma, pi, theta, network, N, K) {
    .Call('_ergmclust_ELBO_conv_HMM_stat_undir', PACKAGE = 'ergmclust', gamma, pi, theta, network, N, K)
}

grad_HMM_stat_undir_K1 <- function(theta, network, N) {
    .Call('_ergmclust_grad_HMM_stat_undir_K1', PACKAGE = 'ergmclust', theta, network, N)
}

hess_HMM_stat_undir_K1 <- function(theta, N) {
    .Call('_ergmclust_hess_HMM_stat_undir_K1', PACKAGE = 'ergmclust', theta, N)
}

ELBO_conv_HMM_stat_undir_K1 <- function(theta, network, N) {
    .Call('_ergmclust_ELBO_conv_HMM_stat_undir_K1', PACKAGE = 'ergmclust', theta, network, N)
}

