# AdamNetCode

Experimentation with AdaNet: Adaptive Structural Learning of Neural Networks.

AdaNet defines an algorithm that adaptively grows a neural network as an ensemble of subnetworks that minimizes the AdaNet objective.

In this project written for the Insight, we study the impact in the loss, produced by the combined weighting of the network ensemble using uniform weights, and alternativley using convex optimization to calculate the optimum weights, and simulateneously control this effect with the regularizing parameter $\lambda$.

$$F(w) = \frac{1}{m} \sum_{i=1}^{m} \Phi \left(\sum_{j=1}^{N}w_jh_j(x_i), y_i \right) + \sum_{j=1}^{N} \left(\lambda r(h_j) + \beta \right) |w_j| $$

where $w$ is the set of mixture weights, one per subnetwork $h$, $\Phi$ is a surrogate loss function such as logistic loss or MSE, $r$ is a function for measuring a subnetwork's complexity, and $\lambda$ and $\beta$ are hyperparameters.


Notes:
How AdaNet uses the objective:

This objective function serves two purposes: To learn to scale/transform the outputs of each subnetwork $h$ as part of the ensemble. To select the best candidate subnetwork $h$ at each AdaNet iteration to include in the ensemble. Effectively, when learning mixture weights $w$ AdaNet solves a convex combination of the outputs of the frozen subnetworks $h$. For $$\lambda \gt 0,$$ AdaNet penalizes more complex subnetworks with greater L1 regularization on their mixture weight, and will be less likely to select more complex subnetworks to add to the ensemble at each iteration.

We will solve a regression task known as the Boston Housing dataset to predict the price of suburban houses in Boston. There are $13$ numerical features, the labels are in thousands of dollars, and 506 examples.
