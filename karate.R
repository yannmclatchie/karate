# load libraries
library(igraph)
library(ggplot2)

# read the karate graph
karate <- make_graph("Zachary")

# plot the club
plot(karate, layout = layout_in_circle)

# convert the graph into an adjacency matrix
adj <- as_adj(karate, sparse = FALSE)

# define model data
data <- list(
  N = vcount(karate), 
  K = 2, 
  beta = c(0.5, 0.5), 
  alpha = rep(1 / 2, 2), 
  graph = adj
)

# fit Bayesian SBM model in Stan
model <- rstan::stan_model(file = "sbm.stan")
fit <- rstan::sampling(model, data = data)
fit
bayesplot::mcmc_areas(
  fit, 
  pars = c("phi[1,1]", "phi[1,2]", "phi[2,1]", "phi[2,2]", "pi[1]", "pi[2]")
)

# extract community membership
fit_ss <- rstan::extract(fit, permuted = TRUE)
membership <- apply(fit_ss$clusters_inf, 2, median)
communities <- list("1" = which(membership == 1), "2" = which(membership == 2))
communities

# define true memberships
true_membership <- c(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 
                     2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)

# plot clusters
shapes <- ifelse(true_membership == 1, "circle", "square")
colours <- ifelse(true_membership == 1, "lightyellow", "lightgreen")
plot(karate, 
     col = membership,
     mark.groups = communities,
     vertex.shape = shapes,
     vertex.color = colours,
     layout = layout_in_circle)

# prior sensitivity
pss <- priorsense::powerscale_sequence(fit)
priorsense::powerscale_plot_ecdf(
  pss, 
  variables = c(
    "phi[1,1]", "phi[1,2]", "phi[2,1]", "phi[2,2]", "pi[1]", "pi[2]"
  )
) + theme_classic() + theme(title = element_blank())
