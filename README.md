# TODO
Am besten abhacken und Namen daneben schreiben wenn man eine Sache abgeschlossen hat
- [x] Some Task... @Arwin

### General Questions
- [ ] why do you think that such a “neural operator for data interpolation” allows for the encoding of local features in time series data?
- [ ] is it really an operator what we are trying to learn? 

### Synthetic Dataset
- [x] Sample a set of 100K random one-dimensional, target interpolation functions “f(t)” on the interval [0, 1], defined on a fine grid of 128 points. Sample them from GPs with an RBF kernel.
      This kernel has a single hyperparameter, which controls the scale at which variations take place. You could use e.g. a Beta distribution over the values of this scale hyperparameter,
      as to increase the variety of the vector fields in your dataset. Note that one of the key aspects of “zero-shot” inference is precisely the variability of the training set. @Arwin
  - [x] **General question 3**: why? @Arwin
- [x] Sample random observation grids (aka sensors). These grids correspond to the observation times at which your hidden function f(t) is observed, and thus define the input of the inference model (see below).
      In other words, each grid can be viewed as a subset of the 128-point fine grid of values of f(t). Feel free to define these grids in any way you see fit. They can be regular or irregular (in time), or mixture thereof.
      They should also have different numbers of points each. In fact, do control the number of minimum points on these grids. One way to implement them in practice would be via random masks. @Arwin
- [x] Since your model is supposed to handle any observation sequence, regardless of the scale of the functions involved, take care of the normalization of the function in your datasets. Check out (and modify, if needed) instance normalization. @Arwin
- [x] In real world applications the data is noisy. We should therefore simulate such noise signals. We shall consider additive Gaussian noise of zero mean,
      because it is the most likely noise family given that we typically only measure the scale of our errors. @Arwin
   - [ ] **General question 4:**  Why?
   - [x] Sample the standard deviation (std) of each noise process (one per function f(t)) from a second Gaussian with zero mean and std = 0.1. @Arwin
- [ ] Finish Synthetic Dataset Class

### Inference Model

- [ ] your model should take as input "time series tuples" like

      (y1, t1), (y2, t2), …, (yL, tL), 
     where t1, …, tL correspond to an instance of your random observation grids, and y1, …, yL correspond to noisy values of the interpolation function f(t) at those observation times.
- [ ] Use a Transformer network to process the input data. You can also consider using time and value embeddings too.
- [ ] Use the ideas of [DeepONet](https://arxiv.org/abs/1910.03193) framework to estimate the "hidden" function f(t).

### Out-of-distribution

- [ ] Finally, we would like you to prepare a short script or notebook that loads your pretrained model and has some plotting function and RMSE function coded in.<br>
      We will test your model on the fly (that is, during your presentation) on one “out-of-distribution” function.<br>
      We will provide you with one noisy time series which has more than 8 and less than 128 noisy tuples of the form: (y1, t1), … (yL, tL).<br>
      We will ask you to plot the estimated interpolation function on some target grid and compute the RMSE wrt the ground-truth function (which we will also provide)
