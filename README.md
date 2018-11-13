# AdamNetCode

Experimentation with AdaNet: Adaptive Structural Learning of Neural Networks.

AdaNet defines an algorithm that adaptively grows a neural network as an ensemble of subnetworks that minimizes the AdaNet objective.

In this project written for Insight, we study the impact in the loss, produced by the combined weighting of the network ensemble using uniform weights, and alternativley using convex optimization to calculate the optimum weights, and simulateneously control this effect with the regularizing parameter.


How AdaNet uses the objective:

This objective function serves two purposes: To learn to scale/transform the outputs of each subnetwork h as part of the ensemble. To select the best candidate subnetwork at each AdaNet iteration to include in the ensemble. Effectively, when learning mixture weights AdaNet solves a convex combination of the outputs of the frozen subnetworks h.  As the value of the regularization parameter increases, AdaNet penalizes more complex subnetworks with greater L1 regularization on their mixture weight, and will be less likely to select more complex subnetworks to add to the ensemble at each iteration.
