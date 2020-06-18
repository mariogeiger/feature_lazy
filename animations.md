## Feature vs Lazy Learning: *the Stripe Model*

**Reference:** [article][1]

We consider a binary classification task where the label function 
<img src="https://render.githubusercontent.com/render/math?math=y(\vec x)">
depends only on one direction in the data space, namely 
<img src="https://render.githubusercontent.com/render/math?math=y( \vec x)=y(x_\parallel)">.
Layers of <img src="https://render.githubusercontent.com/render/math?math=y=+1"> and 
<img src="https://render.githubusercontent.com/render/math?math=y=-1">
regions alternate along the direction <img src="https://render.githubusercontent.com/render/math?math=x_\parallel">
, separated by parallel planes. The two labels are assumed equiprobable. The points 
<img src="https://render.githubusercontent.com/render/math?math=\vec x"> that constitute the training and test set are iid of distribution 
<img src="https://render.githubusercontent.com/render/math?math=\rho(\vec x) = \rho_\parallel(x_\parallel)\rho_\bot(x_\bot)">, where all <img src="https://render.githubusercontent.com/render/math?math=\rho_j">
are drawn from a Gaussian 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}(0,1)">
.

We take a fully-connected one hidden neural network of activation <img src="https://render.githubusercontent.com/render/math?math=\sigma">,

<img src="https://render.githubusercontent.com/render/math?math=f(\vec x) = \frac{1}{h} \sum_{n=1}^h \beta_n \: \sigma\left(\frac{\vec \omega_n \cdot   \vec x}{\sqrt{d}} + b_n\right)">

and we train the function 
<img src="https://render.githubusercontent.com/render/math?math=F(\vec x) = \alpha \left(f(\vec x) - f_0(\vec x)\right)">
with a discrete approximation of Gradient Flow

<img src="https://render.githubusercontent.com/render/math?math=\dot{W} = -\partial_W \frac{1}{p}\sum_\mu l\left(y^\mu F(\vec x^\mu)\right)">. 

Varying the scale 
<img src="https://render.githubusercontent.com/render/math?math=\alpha">
drives the network dynamics from the feature regime (small 
<img src="https://render.githubusercontent.com/render/math?math=\alpha">
) to the lazy regime (large 
<img src="https://render.githubusercontent.com/render/math?math=\alpha">
).

In the following animation we show the evolution of the vectors 
<img src="https://render.githubusercontent.com/render/math?math=\beta_n \vec \omega_n">
during training in the *feature regime*

<p align="center">
  <img width="350" height="350" src="https://github.com/leonardopetrini/feature_lazy/blob/experimental/stripe_wbeta_wlegend.gif">
</p>

*Note*: considering weights magnitude exploses during learning, vectors length is divided by 
<img src="https://render.githubusercontent.com/render/math?math=\max(|\beta|\: ||\vec \omega||)">
in order to make them fit in the frame. Relative norms and orientations are the quantities of interest.

*Parameters*: 
* Training set size: <img src="https://render.githubusercontent.com/render/math?math=p = 1000">
* <img src="https://render.githubusercontent.com/render/math?math=\alpha = 10^{-6}">
* Number of neurons: <img src="https://render.githubusercontent.com/render/math?math=h = 10000">
* Activation function: <img src="https://render.githubusercontent.com/render/math?math=\sigma(\cdot) = ReLU(\cdot)">
* Hinge loss: <img src="https://render.githubusercontent.com/render/math?math=l(z) = \max(0, 1 - z)">
* All weights are initalized <img src="https://render.githubusercontent.com/render/math?math=\vec \omega_n, \b_n, \beta_n \sim \mathcal{N}(0,\mathbf{1})">
* <img src="https://render.githubusercontent.com/render/math?math=f_0"> is the network function at initialization.

[1]:https://
