//
// This Stan program defines a stochastic block model (SBM).
//

data{
	int<lower=1> N; // number of nodes
	int<lower=1> K; // the number of communities
	vector[K] alpha; // Dirichlet parameter
	vector[2] beta; // Beta parameters
	int graph[N, N]; // graph adjacency matrix
}

parameters{
	matrix<lower=0, upper=1>[K,K] phi; // block matrix
	simplex[K] pi; // community membership
}

model {
  for(i in 1:K){
    for(j in 1:K){
      phi[i][j] ~ beta(beta[1], beta[2]); // prior on block matrix entries
    }
  }

  pi ~ dirichlet(alpha); // mixture distribution

  for(i in 1:N){
    for(j in i+1:N){ // symmetry and ignore diagonals

      // likelihood
      graph[i][j] ~ bernoulli(pi' * phi * pi); 
    }
  }
}

generated quantities{
  // log-likelihood
  real log_lik = 0.0;

  matrix[N,K] log_zprob; //cluster probability of each node
  vector[K] lps;
  int clusters_inf[N];

  for(i in 1:N){
    for(j in i+1:N){

      //marginalize out clusters
      log_lik += bernoulli_lpmf(graph[i][j]|pi' * phi * pi);
    }
  }

  // clusters
  for(i in 1:N){

    clusters_inf[i] = 1;
    for(z_i in 1:K){
      log_zprob[i][z_i] = log(pi[z_i]);

      for(j in 1:N){

        for(z_j in 1:K){
          lps[z_j] = log(pi[z_j]) + bernoulli_lpmf(graph[i][j] | phi[z_i][z_j]);
        }

        log_zprob[i][z_i] += log_sum_exp(lps);
      }

      if(log_zprob[i][z_i] > log_zprob[i][clusters_inf[i]]){
        clusters_inf[i] = z_i;
      }
    }
  }
  
  // log prior
  real lprior;
  // joint prior specification
  lprior = dirichlet_lpdf(pi | alpha);
  for(i in 1:K){
    for(j in 1:K){
      lprior += beta_lpdf(phi[i][j] | beta[1], beta[2]); //prior on block matrix entries
    }
  }
}
