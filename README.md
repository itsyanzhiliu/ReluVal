# ReluVal
implement [Formal Security Analysis of Neural Networks using Symbolic Intervals](https://arxiv.org/pdf/1804.10829.pdf) in Python.

## Algorithm 1
Symbolic Interval Propagation is a technique introduced in the paper to mitigate the input dependency problem and tighten the output interval estimation in Deep Neural Networks (DNNs) . It involves keeping symbolic equations throughout the intermediate computations of a DNN, which helps eliminate input dependency errors in the case of linear transformations. However, when passing an equation through a Rectified Linear Unit (ReLU) node, the equation is dropped and replaced with 0 if it can evaluate to a negative value for the given input range. To address this, the technique keeps the lower and upper bound equations for as many neurons as possible and only concretizes as needed.
This approach helps in accurately bounding linear transformations and handling non-linearity in DNNs.

## Algorithm 2
The Backward propagation algorithm approximates the influence caused by ReLUs in the target Deep Neural Network. It works with intervals and computes the output ranges for each node by leveraging symbolic interval analysis. The algorithm uses the inclusion isotonicity properties of the DNN to compute rigorous bounds on the DNN outputs. It applies interval hadamard product and symbolic propagation to minimize overestimations of output bounds. The backward computation of input feature influence is performed by approximating the influence caused by ReLUs.

## Algorithm 3
Iterative Interval Refinement is used to refine the estimated output intervals for complex neural networks, especially when the input intervals are large and result in many concretizations. The algorithm involves repeatedly splitting the input intervals until the output interval is tight enough to meet the desired security property. This iterative bisection process creates a bisection tree, where each bisection on one input yields two children representing two consecutive sub-intervals. The union of these sub-intervals computes the output bound for their parent. This process continues until the output interval is sufficiently tight or a timeout occurs.
