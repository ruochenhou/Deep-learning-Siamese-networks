# Deep-learning-Siamese-networks
In theory, Siamese networks are a kind of neural architecture that measures how different two images are. The inputs for the Siamese Network are a pair of images. The images are positive if they are similar and their target value will be set to 0. On the other hand, the images are negative if they are different and their target value will be set to 1.
Each image is processed by exactly the same base_neural network and these subnetworks share the same weight. The output of the base_neural network are two vectors that will be the input to a layer which measures the distance between those two images.
Siamese Networks can be implemented to recognize faces or handwriting or to index documents among other applications.
