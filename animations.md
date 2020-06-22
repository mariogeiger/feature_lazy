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
<img src="https://render.githubusercontent.com/render/math?math=\rho(\vec x) = \rho_\parallel(x_\parallel)\rho_\bot(x_\bot)">, where all <img src="https://render.githubusercontent.com/render/math?math=\rho">
are standard Gaussian p.d.f's 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}(0,1)">
.

We take a fully-connected one hidden neural network of activation <img src="https://render.githubusercontent.com/render/math?math=\sigma">,

<img src="https://render.githubusercontent.com/render/math?math=f(\vec x) = \frac{1}{h} \sum_{n=1}^h \beta_n \: \sigma\left(\frac{\vec \omega_n \cdot   \vec x}{\sqrt{d}} %2B b_n\right)">

and we train the function 
<img src="https://render.githubusercontent.com/render/math?math=F(\vec x) = \alpha \left(f(\vec x) - f_0(\vec x)\right)">
with a discrete approximation of Gradient Flow

<img src="https://render.githubusercontent.com/render/math?math=\dot{W} = -\partial_W \frac{1}{p}\sum_\mu l\left(y^\mu F(\vec x^\mu)\right)">. 

Varying the scale 
<img src="https://render.githubusercontent.com/render/math?math=\alpha">
drives the network dynamics from the *feature regime* (small 
<img src="https://render.githubusercontent.com/render/math?math=\alpha">
) to the *lazy regime* (large 
<img src="https://render.githubusercontent.com/render/math?math=\alpha">
).

In the following animation we show the amplification effect taking place during the evolution of the vectors 
<img src="https://render.githubusercontent.com/render/math?math=\beta_n \vec \omega_n">
during training in the *feature regime*

<p align="center">
  <img width="350" height="350" src="https://github.com/leonardopetrini/feature_lazy/blob/experimental/evolution_stripe_feature.gif">
</p>

The animation is consistent with the fact that the amplification factor
<img src="https://render.githubusercontent.com/render/math?math=\lambda = \sqrt{\frac{\langle \omega_\parallel^2\rangle_h}{\langle\bar\omega_\bot^2\rangle_h}}">
diverges. 

*Note*: considering weights magnitude exploses during learning, vectors length is divided by 
<img src="https://render.githubusercontent.com/render/math?math=\max(|\beta|\: ||\vec \omega||)">
in order to make them fit in the frame. Relative norms and orientations are the quantities of interest.

Considering we choose <img src="https://render.githubusercontent.com/render/math?math=\sigma(\cdot) = ReLU(\cdot)">,
we can plot the point in space, nearest to the origin, for which the ReLU argument is zero. For each neuron, this is given by 
<img src="https://render.githubusercontent.com/render/math?math=-b_n \frac{\omega_n}{||\omega_n||^2}">.

In the following we plot the evolution during learning of
<img src="https://render.githubusercontent.com/render/math?math=-b_n \frac{\omega_n}{||\omega_n||^2}">
for each neuron. Points are colored depending on 
<img src="https://render.githubusercontent.com/render/math?math=sign(b_n)">
which says if the ReLU function is oriented towards the origin or away from it
.
<p align="center">
  <img width="700" height="350" src="https://github.com/leonardopetrini/feature_lazy/blob/experimental/particles_stripe_feature.gif">
</p>


*Parameters*: 
* Training set size: <img src="https://render.githubusercontent.com/render/math?math=p = 1000">
* <img src="https://render.githubusercontent.com/render/math?math=\alpha = 10^{-6}">
* Number of neurons: <img src="https://render.githubusercontent.com/render/math?math=h = 10000">
* Activation function: <img src="https://render.githubusercontent.com/render/math?math=\sigma(\cdot) = ReLU(\cdot)">
* Hinge loss: <img src="https://render.githubusercontent.com/render/math?math=l(z) = \max(0, 1 - z)">
* All weights are initalized <img src="https://render.githubusercontent.com/render/math?math=\vec \omega_n, \b_n, \beta_n \sim \mathcal{N}(0,\mathbf{1})">
* <img src="https://render.githubusercontent.com/render/math?math=f_0"> is the network function at initialization.

[1]:https://
