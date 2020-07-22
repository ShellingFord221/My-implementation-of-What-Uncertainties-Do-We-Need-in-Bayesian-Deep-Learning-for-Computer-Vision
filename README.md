# My-implementation-of-What-Uncertainties-Do-We-Need-in-Bayesian-Deep-Learning-for-Computer-Vision
This is my implementation of classification task in paper _What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision_.

In this repo, I model aleatoric uncertainty, epistemic uncertainty and both of them of a classification task with MNIST dataset. I also implement a normal network to be compared with.

However, these are all based on my understanding of this paper. In regression task, it is easy to compute `var(mean)` (i.e. epistemic uncertainty) and `mean(var)` (i.e. aleatoric uncertainty), but in classification task, I really don't know how to compute aleatoric uncertainty and epistemic uncertainty for each sample, since they are all vectors rather than a single value. If I can compute them, I also don't know how to plot these uncertainties like regression task. (**New**: After reading so many papers, I think the best way to quantify uncertainties in classification task is entropy, though it is not comparable with the measurement of variance in regression task. In this situation, the `sigma` vector **is just used in the loss function to mitigate the influence of noisy samples**. Explicit quantification of uncertainties is better done by the entropy of `mu`. I'll implement this later.)

~~Besides, I still can't make sure that whether each logit value is drawn from a Gaussian and the whole logit vector is drawn from a multi-dimensional Gaussian distribution. I saw other repos predict variance of each sample by a single value, while I think it should be a vector hence the variance is a 'diagonal matrix with one element for each logit value'.~~ (This is discussed in issue #1)

The results seem that modeling aleatoric uncertainty can improve model's performance.

Any feedback and discussion is welcome.
