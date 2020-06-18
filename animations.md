## Feature vs Lazy Learning: The Stripe Model

**Reference:** [article][1]

We consider a binary classification task where the label function 
<img src="https://render.githubusercontent.com/render/math?math=y(\vec x)">
depends only on one direction in the data space, namely 
<img src="https://render.githubusercontent.com/render/math?math=y( \vec x)=y(x_1)">.
Layers of <img src="https://render.githubusercontent.com/render/math?math=y=+1"> and 
<img src="https://render.githubusercontent.com/render/math?math=y=-1">
regions alternate along the direction <img src="https://render.githubusercontent.com/render/math?math=x_1">
, separated by parallel planes. The two labels are assumed equiprobable. The points 
<img src="https://render.githubusercontent.com/render/math?math=\vec x"> that constitute the training and test set are iid of distribution 
<img src="https://render.githubusercontent.com/render/math?math=\rho(\vec x) = \rho_1(x_1)\cdots\rho_d(x_d)">, where all <img src="https://render.githubusercontent.com/render/math?math=\rho_j">
are continuous, of zero mean and of similar spatial scale. Also, the distribution <img src="https://render.githubusercontent.com/render/math?math=\rho_1">
does not vanish at the location of the interfaces (no margin).

We take a fully-connected one hidden neural network of activation <img src="https://render.githubusercontent.com/render/math?math=\sigma">,

<img src="https://render.githubusercontent.com/render/math?math=f(\vec x) = \frac{1}{h} \sum_{n=1}^h \beta_n \: \sigma\left(\frac{\vec \omega_n \cdot   \vec x}{\sqrt{d}} + b_n\right)">

and we train the function 
<img src="https://render.githubusercontent.com/render/math?math=F(\vec x) = \alpha \left(f(\vec x) - f_0(\vec x)\right)">
with a discrete approximation of Gradient Flow:

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
during training


![6pt2_gif](https://github.com/leonardopetrini/feature_lazy/blob/experimental/stripe_wbeta_wlegend.gif)

*Note*: considering weights magnitude exploses during learning, vectors length is divided by 
<img src="https://render.githubusercontent.com/render/math?math=\max(|\beta|\: ||\vec \omega||)">
in order to make them fit in the frame. Relative norms and orientations are the quantities of interest.


[1]:https://
