printed-digits-classification
=============================

This project presents a hybrid algorithm for training RBF network based on K-means and SOM. The algorithm consists of a proposed clustering algorithm to position the RBF center and givens least squares to estimate the weights. The aim of this experiment is to recognize printed digits (1-4) using the hybrid model. In the meanwhile, KNN and MLP with Scaled Conjugate Gradient will be implemented in order to show the comparative of different models according to the experiments.


Howto
-----
1.use matlab to add path.

2.defualt repeattime is 1,you can modify the rts premeter of the code.

3.in the command window,type the name(im_rbf_som,im_rbf_kmeans,im_mlpscg or im_knn).

4.press enter,the classification process would begin ,after it finished,the accuracy and running time would be displayed.
