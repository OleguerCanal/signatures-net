# Signet Refitter

![image](https://user-images.githubusercontent.com/31748781/192327635-d5001335-1780-4b74-8e01-8b6fb3c95228.png)

The SigNet Refitter module gets as an input a set of mutational vectors and as an output it predicts a guess on the weight presence of each mutational signature with an associated error interval. It is composed by 4 main modules: [SigNet Detector](signet_detector.md), Non-negative least-squares, Finetuner and Error Estimator.

## Non-negative least-squares (NNLS)

In mathematical optimization, the problem of non-negative least squares (NNLS) is a type of constrained least-squares problem in which the coefficients are not allowed to become negative. In the studied case, this ensures that the solutions are biologically feasible.
By applying the NNLS optimization we are minimizing the Euclidean distance between a given sample profile and the reconstructed profile.
This is what other algorithms use to find the weights of the decomposition.
However, when the number of mutations in the sample is low, such as in WES, RNA sequencing data or even WGS for some cancer types, the input profile does not look like the real profile that corresponds to the linear combination of signatures due to the high degree of stochasticity.
Thus, we hypothesize that this kind of classical algorithm cannot be used in the regime of few mutations and we turn to using artificial neural networks instead.

## Finetuner

The FineTuner is the neural network of SigNet Refitter in charge of finding the proper signature decomposition.
For this, we do not have a single neural network but, instead, we have two neural networks for two different regimes: "low" and "high" number of mutations.
We make this distinction since low-mutational profiles are dominated by noise and the reconstructed profiles might not look like the input profiles.
Here, correlations between signatures are the most important feature that the neural network can use to predict the signature weights.
However, for high numbers of mutations the reconstructed profiles should be close to the inputs and this can be used by the neural network to decompose the samples.
The input of each neural network is different:
- For the **large** number of mutations network we input three different objects: we provide the normalized mutation vector, the number of mutations of the sample, and the baseline guess that we obtained from the NNLS step.
The latter functions as a starting guess for the neural network that should be fine-tuned to find a better decomposition.

- The **low** number of mutations neural network gets inputted the normalized mutation vector and the number of mutations.
The NNLS guess is not helpful in this regime.
The output of both neural networks is the same: the signature decomposition of the sample which correspond to a vector of length 72 containing the weight of each of the signatures in COSMIC v3.1. 

The training set for this neural network is composed of realistic looking samples generated by sampling from the real linear combination of signatures extracted from the PCAWG dataset. For each sample we generated ten samples with different numbers of mutations ranging from 25 mutations to the order of $10^5$.

The FineTuner loss that we used to train the neural network is based on the distance of two categorical probability distributions of signatures, the Kullback-Leibler divergence

## ErrorFinder

The weight decomposition that we find after the FineTuner step is not exact. Its accuracy will depend on both the number of mutations of the sample and the signatures that are present. Namely, when the number of mutations is low, the sample mutational profile is more stochastic and it is more difficult to decompose than that of a sample with high mutation count. Therefore, we would expect to have a less accurate weight decomposition for samples with lower than with higher mutation count. Furthermore, there are some signatures which, because of their shape (i.e. they are flat and do not have specific traits that make them distinguishable), can be more difficult to identify, implying that their weight predictions are less accurate.
We want to be able to quantify these different sources of error by finding prediction intervals with a certain confidence level per signature and sample. As far as we are aware, there is little known about prediction intervals for neural networks. We first tried to apply the same architecture of the only article we found that discusses this topic. However, training our network based on that approach failed and we instead developed a modified method that would work for our inference.

In order to find prediction intervals there are two quantities that need to be minimized at the same time: the width of the intervals and the distance between the real value and the interval. The former is necessary so that the intervals have a reasonable width. If we do not impose this, the neural network learns to create intervals that cover the whole range between 0 and 1, making the intervals uninformative. The latter is necessary in order to make sure that some proportion of the real values fall inside the interval. We want them to be as small as possible while containing most of the real values inside of their range (ideally at least 95\% in order to have this level of confidence).
