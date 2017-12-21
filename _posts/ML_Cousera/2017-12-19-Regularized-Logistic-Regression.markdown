---
layout: "post"
title:
---

From our previous post, we understand that most real-world problems are not linearly separable, which means our dataset cannot be separated into positive and negative examples by a straight-line through a plot. Now, let us take a look on how this issue of be addressed.

We will implement a regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA) using historical data. During QA, each microchip goes through various tests to ensure it is functioning correctly. Based on our historical data, let us build our logistic regression model.

Back to the previously mentioned non-linearly separable problem. Figure 1 shows an example of a problem of such

{: .center}
![image]({{ site.baseurl }}/public/project-images/regularized_logistic_regression/logistic_regression.png)
*Figure 1: Visualizing the scatter plot*

Since we know that a straight-forward application of logistic regression will not perform well on this dataset, we need to create more features from each data point where we will map the features into all polynomial terms of $$x_1$$ and $$x_2$$ up to the sixth power

$$mapFeature(x)=1,x_1,x_2,x_1^2,x_1x_2,...,x_1x_2^5, x_2^6$$

As a result of this mapping, we will be able to transform our vector into a 28-dimensional vector. A logistic regression classifier trained on this high-dimension feature vector will have a more complex decision boundary and will appear non-linear when drawn in our 2-dimensional plot. While the feature mapping allows us to build a more comprehensive classifier, it is also more susceptible to **over-fitting** (opposite of **under-fitting**). Let us compare the difference between them.

| Over-fitting 	| Under-fitting 	|
|---------------------------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| Also known as High Variance 	| Also known as High Bias 	|
| Caused by a hypothesis function $$h$$ that fits the available data but does not generalize well to predict new data 	| Happens when the form of our hypothesis function $$h$$ maps poorly to the trend of the data. 	|
| Usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data. 	| Usually caused by a function that is too simple or uses too few features. eg. if we take $$h_\theta(x)=\theta_0+\theta_1x_1+\theta_0x_2$$, then we are making an initial assumption that a linear model will fit, the training data well and will be able to generalize but that may not be the case. 	|



One way to overcome over-fitting is through **regularization**, where what is actually done is just merely reducing the parameters $$\theta_j$$ while keeping all the features. To achieve, we introduce our regularization parameter, $$\lambda$$. The regularized cost function in logistic regression is as follows

$$J(\theta)=\frac{1}{m}\sum_{i=1}^m [-y^{(i)}\log(h_\theta(x^{(i)}))-(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$$


**How to overcome underfit/overfit**