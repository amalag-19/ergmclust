\name{hess_HMM_stat_undir}
\alias{hess_HMM_stat_undir}
\title{
C++ function for updating the Hessian matrix in a static binary undirected network.
}
\description{
The C++ function hess_HMM_stat_undir() updates the Hessian matrix when the number of clusters is greater than or equal to 2 in a static binary undirected network.
}
\usage{
hess_HMM_stat_undir(theta, gamma, N, K)
}
\arguments{
  \item{theta}{
  Vector of current \code{theta} (network canonical parameters) values.
  }
  \item{gamma}{
  Vector of current \code{gamma} (variational parameters) values.
  }
  \item{N}{
  Number of nodes.
  }
  \item{K}{
  Number of clusters.
  }
}
\value{
Returns the Hessian matrix of ELBO with respect to theta parameters for Newton-Raphson update in M-step.
}
\keyword{internal}