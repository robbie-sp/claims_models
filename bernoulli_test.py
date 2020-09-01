#%%
import os
from cmdstanpy import cmdstan_path, CmdStanModel

bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)
bernoulli_model.name
bernoulli_model.stan_file
bernoulli_model.exe_file
print(bernoulli_model.code())

bernoulli_data = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')
bern_fit = bernoulli_model.sample(data=bernoulli_data, output_dir='./model_output')

print(bern_fit)

print(bern_fit.sample.shape)

print(bern_fit.summary())

print(bern_fit.diagnose())