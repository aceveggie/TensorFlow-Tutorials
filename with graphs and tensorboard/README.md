I wrote a small Tensorflow program and monitored its progress on Tensorboard.


**Tensorboard** is for visualization purposes.
You can track scalar values (cost, accuracy, etc.), arrays/tensors (weights, biases).
You can also compute mean, std-dev, plot histogram of values, etc. This will help us understand what the network is learning and what the weights consist of.

More importantly, you can also use this to compute **TSNE and PCA** of the data towards the end of the network and visualize it in 2D-3D. This will be useful if you want to learn how effectively the network is able to differentiate between classes.



**Here you can see the cost over time:**

![Cost over time](./images/cost_over_time_from_tensorboard.png)





**Here you can see the accuracy:**

![Accuracy over time](./images/accuracy_from_tensorboard.png)
