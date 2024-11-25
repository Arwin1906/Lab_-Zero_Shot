# TODO
Am besten abhacken und Namen daneben schreiben wenn man eine Sache abgeschlossen hat
- [x] Some Task... @Arwin

## ----------------------- * 2nd Synthetic Dataset* ----------------------
- [ ] Sample a set of 100K random one-dimensional, target functions “f(t)” on the interval [0, 1], defined on a fine grid of 640 points.
- [ ] Sample the target functions from GPs with: periodic (30), locally periodic (30) and linear plus periodic or locally periodic (40), whose definition you can find in: https://www.cs.toronto.edu/~duvenaud/cookbook/

Try imposing distributions on the parameters of these kernels to increase their variance. The numbers in parenthesis are some tentative proportion. Feel free to change these.<br>
- [ ] As before, sample random observation grids (aka sensors). You can implement these via the attention masks of your transformer.
- [ ] Corrupt the time series with additive Gaussian noise, just as before
- [ ] Split your fine grid of time series into **5 sets**, of 128 points each. Note that you can split your attention masks equivalently. This means that each set (or window) spans 20% of the (continuous) domain of your target functions.
- [ ] Preprocess each of these windows with your pretrained branch net from task 1.
- [ ] Remember that your pretrained model only processes normalized series. Therefore, normalize your windows accordingly, but **keep these normalization values**, they will be used by your second transformer network

## ----------------- Inference Model 1 [IM1] (Modifications to your model from task 1) -----------------

Your model from task one consisted of a trunk and a branch net. The branch net, let's call it Transformer1, processed your input series and returns as many outputs as inputs:

$h_1,..., h_L = Transformer1[(y_1, t_1), (y_2, t_2), …, (y_L, t_L)]$   (where L=128 during training)
Now consider a summary network that maps the set h1, …, hL onto a single embedding thus:
$h_b = Attention(q, W_k . H, W_v . H)$, where:

- [ ] H is the matrix of your L embeddings, i.e. $H = h_1,..., h_L,$
- [ ] $W_k$ and $W_v$ and the learnable key and value projections, and
- [ ] q is a learnable embedding (the learnable query).

Before you followed DeepONet and used a dot product between the outputs of the trunk and branch nets. Let’s now upgrade this dot product with an MLP-out. That is, call $h_t$ = T(eval-time) the output of the trunk net.

Then the output of your complete model **IM1** is: $f(t) = MLP_{out}(h_t, h_B)$ <br>
Retrain this modified architecture on your synthetic dataset from task 1

## ---------------------- Inference Model 2 [IM2]  ----------------------

Once you (re)trained IM1 above, you can use it to encode the local fluctuations of your long 640 time series.

- [ ] For each time series of your second dataset you can now construct $h_{b1}, h_{b2}, h_{b3}, h_{b4}, h_{b5}$ local embeddings
- [ ] Similarly you can construct a sequence of normalizations (or statistics) per time series: $s1, s2, s3, s4, s5$.
- [ ] Embed these statisticswith an MLP and concatenate them with your local embeddings so that:
$h_{bm} <-$ Concat($h_{bm}, MLP(s_m)$), for m = 1, 2, 3 and 4.

- [ ] Define a second transformer so that: $u1, u2, u3, u4 =$ Transformer2($h_{b1}, h_{b2}, h_{b3}, h_{b4}$) and note that we only use the first 4 embeddings.
- [ ] Define a second summary network (with a second learnable query) to obtain a single embedding u summarizing $u_1, .., u_4$.
- [ ] Use your summary u together with the pretrained MLP-out from IM1 to compute f(t) on the last window
- [ ] Train your model to reconstruct the clean function f(t) on that last window (note that you can also use $h_{b5}$ somehow) **while keeping the parameters of IM1 frozen**

## --------------------- Evaluation ---------------------

- [ ] Try to find the ETTh1 (horizon=96) dataset, as used by https://proceedings.mlr.press/v235/das24c.html in Table 5, and test your model on it
- [ ] Also prepare your notebook as before. We will give you some target datasets to evaluate on the fly during our meeting
