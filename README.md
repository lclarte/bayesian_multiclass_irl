## Summary

This project deals with the problem of bayesian multitask inverse reinforcement learning (or Bayesian MT-IRL) for Partially Observable Markov Decision Processes (POMDP) : consider an agent whose policy is parametrized by a vector _w_, whose prior distribution is a mixture of multivariate Gaussian, where the parameters <img src="https://render.githubusercontent.com/render/math?math=\mu, \Sigma"> of the Gaussian components <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}(\mu, \Sigma)"> are sampled from a Normal Inverse Wishart hyperprior. The goal is to infer the posterior distribution of the <img src="https://render.githubusercontent.com/render/math?math=\mu, \Sigma"> observing the trajectories generated in the MDP. 

Relevant articles : 
* [Lazaric et al., Bayesian multitask reinforcement learning](https://hal.inria.fr/inria-00475214/document)
* [Dimitrakakis et al., Bayesian multitask inverse reinforcement learning](https://arxiv.org/abs/1106.3655)
* [Choi et Kim, Nonparametric bayesian inverse reinforcement learning for multiple reward functions](https://papers.nips.cc/paper/4737-nonparametric-bayesian-inverse-reinforcement-learning-for-multiple-reward-functions)

## Requirements

This code requires https://github.com/IDSIA/sacred

## Code organization 

There are 4 main folders : _core_, _experiments_, _envs_ and _test_. 
* _core_ : contains main algorithms used for the inference. 
* _test_ : unitary test for functions in _core_ 
* _envs_ : definitions of the RL environments used for our experiments
* _experiments_ : contains .py files, one for each experiment, and _logs_ folder to store results 

## How to use

This project uses the [sacred library](https://github.com/IDSIA/sacred). This allows the user to tune parameters in the CLI. 

### Running unitary tests 

In the root folder, run the module unittest on the desired python script. example : 
```
python -m unittest test.test_bp
```

### Running experiments

In the root folder, simply run the desired python script :
```
python -m experiments.exp_chain
```
You can tune some hyperparameters such as the number of MDPs to infer, the length of trajectories, etc. by setting them in the CLI. For example, if you want trajectories length = 10 and number of MDPs = 20, simply run 
```
python -m experiments.exp_chain with T=10 M=20
```



