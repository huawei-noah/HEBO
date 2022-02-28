## IP Video script

Hello everyone,
I'm Antoine and I'm gonna present you BOiLS: Bayesian Optimisation for Logic Synthesis, a paper accepted at DATE22 conference.

Let's start with the problem definition. And-inverter graphs (known as AIGs) are used to represent the logical functionality of circuits. A given AIG can be made more compact and thus more practical by applying a sequence of pre-defined operators (such as refactor or balance). The goal is to find  the sequence of operators yielding the highest Quality of Results (or QoR), which is a metric capturing the compactness of the obtained AIG. As the function mapping a sequence to a QoR  cannot be easily expressed in a closed form,
current ML methods developed to tackle this optimisation problem require the evaluation of many sequences which is not very efficient.

That is why we introduce a Bayesian optimisation framework to tackle this problem in a sample efficient way. To do so we build a specific surrogate model using a Gaussian Prossess with a kernel suited to compare sequences of operators. We also account for the high dimensionality of the search space by resorting to a local search strategy when looking for the next sequence to evaluate.

To assess our method efficiency, we took 10 circuits from a widely used benchmark and compared BOiLS to many other ML optimisers and heuristics. By observing the evolution of the Quality of Result as the number of evaluated sequences increases we see that in most cases BOiLS find the best sequence among all methods while requiring a way smaller number of evaluations, which opens the path to more efficient logic synthesis.
