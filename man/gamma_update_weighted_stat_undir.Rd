\name{gamma_update_weighted_stat_undir}
\alias{gamma_update_weighted_stat_undir}
\title{
C++ function for updating the variable \code{gamma} (variational parameters) in a weighted static binary undirected network.
}
\description{
The C++ function gamma_update_weighted_stat_undir() updates the value of variable \code{gamma} (variational parameters) in a weighted static binary undirected network.
}
\usage{
gamma_update_weighted_stat_undir(gamma, pi, theta, block_dens_mat, adjmat, N, K)
}
\arguments{
  \item{gamma}{
  Vector of current \code{gamma} (variational parameters) values.}
  \item{pi}{
  Vector of current \code{pi} (mixture proportions) values.
  }
  \item{theta}{
  Vector of current \code{theta} (network canonical parameters) values.
  }
  \item{block_dens_mat}{
  Array of a network density matrices.
  }
  \item{adjmat}{
  Array of a network adjacency matrices.
  }
  \item{N}{
  Number of nodes.
  }
  \item{K}{
  Number of clusters.
}
}
\value{
Returns all quadratic and linear term coefficients for the quadratic programs corresponding to all nodes.
}
\keyword{internal}