#%%
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import cmdstanpy
import os

cmdstanpy.utils.cxx_toolchain_path()

#%% [markdown]
## Generate Claims

# Simple generator of lognormal claims

#%%
# parameters
mu = 0.5
sigma = 1

dist = stats.lognorm(np.exp(mu), sigma)

# generate claims
claims = dist.rvs(size=1000)

# graph
sns.distplot(claims)

#%% [markdown]
# \begin{aligned}
# X_i & \sim LN(mu, sigma) \\
# mu & \sim N( \mu_{mu}, 1 ) \\
# sigma & \sim exp( \lambda_{sigma} ) \\
# \mu_{mu} & = 1 \\
# \lambda_{sigma} & = 1 \\
# \end{aligned}

# model

#%%
print(os.getcwd())

#%%
# priors

# prior predictive checks

mu_mu = 0.5
lambda_sigma = 1

samples = 100

prior_scale = stats.norm(mu_mu, 1).rvs(size=samples) # note stats.expon using scale notation so scale = 1 / lambda
prior_shape = stats.expon(1 / lambda_sigma).rvs(size=samples)

#%%
fig, ax = plt.subplots(1,1)

x = np.linspace(0, 200)

for sample in range(samples):

    prior_dist = stats.lognorm(prior_scale[sample], prior_shape[sample])
    ax.plot(x, prior_dist.pdf(x), 'r-', alpha=0.05)


#%%
# Build Model

file_path = os.path.join(os.getcwd(), 'model_1.stan')

model = cmdstanpy.CmdStanModel(stan_file=file_path)
model.name
model.stan_file
model.exe_file
print(model.code())


#%%
# Sample
stan_data = {
    'N': claims.shape[0],
    'claims': claims.tolist()
}

fit = model.sample(data=stan_data) #, output_dir='./model_output')

print(fit)

print(fit.sample.shape)

print(fit.summary())

print(fit.diagnose())

# diagnostics
# validations

# single dist
# add cat
# add variate
# add time

# %%
