# DNN
* backpropagation
  $y=wx$
  $softmax\ :S_i=\frac{e^{y_i}}{\sum_{j}{y_j}}$
  $cross\ entropy\ loss\ : -\sum_{i}{label_ilogS_i}$
  $因為label為\ onehot\ loss為 -logS_i=-y_i+log\sum_{j}{e^{y_j}}$
  $dw=(-1+\frac{e^{y_i}}{\sum_{j}{e^{y_j}}}x_i)\ \ \ \  j=i$
  $dw=(\frac{e^{y_i}}{\sum_{j}{e^{y_j}}}x_i)\ \ \ \  j\ne i$