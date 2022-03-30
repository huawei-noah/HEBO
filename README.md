# Bayesian Optimisation & Reinforcement Learning Research

This directory contains official implementations for Bayesian optimisation & Reinforcement Learning works developped by Huawei, Noah's Ark Lab. 
- [HEBO: Heteroscedastic Evolutionary Bayesian Optimisation](./HEBO) 
- [T-LBO](./T-LBO)
- [BOiLS: Bayesian Optimisation for Logic Synthesis](./BOiLS)
- [Bayesian Optimisation with Compositional Optimisers](./CompBO)
- [SAUTE RL: Almost Surely Safe RL Using State Augmentation](./SAUTE)

Further instructions are provided in the README 
files associated to each project.

# Bayesian Optimisation Research 

## [HEBO](./HEBO)
<img src="./HEBO/hebo.png" alt="drawing" width="400"/>

Bayesian optimsation library developped by Huawei Noahs Ark Decision Making and Reasoning (DMnR) lab. The <strong> winning submission </strong> to the [NeurIPS 2020 Black-Box Optimisation Challenge](https://bbochallenge.com/leaderboard). 

## [T-LBO](./T-LBO)
<p float="center">
  <img src="./T-LBO/figures/LSBO.png" width="400" />
  <img src="./T-LBO/figures/magnets.png" width="400" /> 
</p>

Codebase associated to: [High-Dimensional Bayesian Optimisation withVariational Autoencoders and Deep Metric Learning](https://arxiv.org/abs/2106.03609)
##### Abstract
We introduce a method based on deep metric learning to perform Bayesian optimisation over high-dimensional, structured input spaces using variational autoencoders (VAEs).
By extending ideas from supervised deep metric learning, we address a longstanding problem in high-dimensional VAE Bayesian optimisation, namely how to enforce
a discriminative latent space as an inductive bias. Importantly, we achieve such an inductive bias using just 1% of the available labelled data relative to previous work,
highlighting the sample efficiency of our approach. 
As a theoretical contribution, we present a proof of vanishing regret for our method. As an empirical contribution, 
we present state-of-the-art results on real-world high-dimensional black-box optimisation problems including property-guided molecule generation.
It is the hope that the results presented in this paper can act as a guiding principle for realising effective high-dimensional Bayesian optimisation.

## [BOiLS: Bayesian Optimisation for Logic Synthesis](./BOiLS)
<p align="center">
    <img src="./BOiLS/results/sample-eff-1.png" alt="drawing" width="500"/>
</p>

Codebase associated to: [BOiLS: Bayesian Optimisation for Logic Synthesis](https://arxiv.org/abs/2111.06178) accepted 
at **DATE22** conference.

##### Abstract
Optimising the quality-of-results (QoR) of circuits during logic synthesis is a formidable challenge necessitating
the exploration of exponentially sized search spaces. While expert-designed operations aid in uncovering effective 
sequences, the increase in complexity of logic circuits favours automated procedures. Inspired by the successes of 
machine learning, researchers adapted deep learning and reinforcement learning to logic synthesis applications. However
successful, those techniques suffer from high sample complexities preventing widespread adoption. To enable efficient 
and scalable solutions, we propose BOiLS, the first algorithm adapting modern Bayesian optimisation to navigate the 
space of synthesis operations. BOiLS requires no human intervention and effectively trades-off exploration versus 
exploitation through novel Gaussian process kernels and trust-region constrained acquisitions. 
In a set of experiments on EPFL benchmarks, we demonstrate BOiLS's superior performance compared to state-of-the-art 
in terms of both sample efficiency and QoR values.


## [Bayesian Optimisation with Compositional Optimisers](./CompBO)

<div style="text-align:center"><img src="./CompBO/image/summary-Best-performance-on-Synthetic-tasks-matern-52-3.png" alt="drawing" width="600"/>


Codebase associated to: [Are we Forgetting about Compositional Optimisers in Bayesian Optimisation?](https://www.jmlr.org/papers/v22/20-1422.html)
 accepted at **JMLR**.
##### Abstract
Bayesian optimisation presents a sample-efficient methodology for global optimisation. Within this framework, a crucial performance-determining
subroutine is the maximisation of the acquisition function, a task complicated by the fact that acquisition functions tend to be non-convex and
thus nontrivial to optimise. In this paper, we undertake a comprehensive empirical study of approaches to maximise the acquisition function. 
Additionally, by deriving novel, yet mathematically equivalent, compositional forms for popular acquisition functions, we recast the maximisation
task as a compositional optimisation problem, allowing us to benefit from the extensive literature in this field. We highlight the empirical 
advantages of the compositional approach to acquisition function maximisation across 3958 individual experiments comprising synthetic optimisation 
tasks as well as tasks from Bayesmark. Given the generality of the acquisition function maximisation subroutine, we posit that the adoption of
compositional optimisers has the potential to yield performance improvements across all domains in which Bayesian optimisation is currently 
being applied.
----
  
# Reinforcement Learning Research
## [SAUTE RL: Almost Surely Safe RL Using State Augmentation](./SAUTE)
Codebase associated to: [SAUTE RL: Almost Surely Safe RL Using State Augmentation](https://arxiv.org/pdf/2202.06558.pdf).
 
##### Abstract
Satisfying safety constraints almost surely (or with probability one) can be critical for deployment of Reinforcement Learning (RL) in real-life applications. For example, plane landing and take-off should ideally occur with probability one. We address the problem by introducing Safety Augmented (Saute) Markov Decision Processes (MDPs), where the safety constraints are eliminated by augmenting them into the state-space and reshaping the objective. We show that Saute MDP satisfies the Bellman equation and moves us closer to solving Safe RL with constraints satisfied almost surely. We argue that Saute MDP allows to view Safe RL problem from a different perspective enabling new features. For instance, our approach has a plug-and-play nature, i.e., any RL algorithm can be "sauteed". Additionally, state augmentation allows for policy generalization across safety constraints. We finally show that Saute RL algorithms can outperform their state-of-the-art counterparts when constraint satisfaction is of high importance.
  
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Codebase Contributors 
<strong> Alexander I Cowen-Rivers, Antoine Grosnit, Alexandre Max Maravel, Aivar Sootla, Taher Jafferjee, Ryan Rhys Griffiths, Wenlong Lyu, Zhi Wang. </strong>

  
