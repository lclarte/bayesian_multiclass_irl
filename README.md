### Summary

This project deals with the problem of bayesian multitask inverse reinforcement learning (or Bayesian MT-IRL) for Partially Observable Markov Decision Processes (POMDP). 

### Code organization 

There are 4 main folders : _core_, _experiments_, _envs_ and _test_. 
* _core_ : contains main algorithms used for the inference. 
* _test_ : unitary test for functions in _core_ 
* _envs_ : definitions of the RL environments used for our experiments
* _experiments_ : contains .py files, one for each experiment, and _logs_ folder to store results 

### How to use

This project uses the [sacred library](https://github.com/IDSIA/sacred). This allows the user to tune parameters in the CLI. 