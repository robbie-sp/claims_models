data {
  int<lower=0> N;
  vector[N] claims;
    // real mu_mu;
    // real<lower=0> lambda_sigma;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
    mu ~ normal(0.5, 2);
    sigma ~ exponential(1);
    claims ~ lognormal(mu, sigma);
}