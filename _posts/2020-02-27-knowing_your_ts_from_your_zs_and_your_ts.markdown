---
layout: post
title:      "Knowing your t's from your z's... and your t's"
date:       2020-02-27 22:54:57 +0000
permalink:  knowing_your_ts_from_your_zs_and_your_ts
---


The past couple weeks of studying for me have involved a lot of chin scratching over probability and statistical tests.  For me, it helped to take a bit of a pause and explain everything to myself intuitively to get a good understanding of the complexity and sometimes cryptic terminology of the subject.  In the hopes of helping others who may benefit from a similar intuitive summarization, I will share mine in this post.  

First, let's take a distribution of measurements, for this post we'll use the loudness of words spoken by the person seated next to you on a train during their phone call.  We will assume that the distribution of loudnesses over the whole conversation to be normal, with most words coming in around the mean, some quieter, and some even louder.  You would ideally store this data as a list of loudness measurements, one for each word spoken, which would quickly accumulate a large sample of data points for analysis, but you are limited by taking measurements from a loudness reading phone app, which is only giving one good reading every minute.  You have recently learned z and t tests, and see that this is a good opportunity to pass some time on this journey by performing statistical tests for fun.

To review:

Both z-tests and t-tests are both used for the same purpose: hypothesis testing.  What they tell you is the probability of getting your observed sample from a certain population, and if this probability turns out to be sufficiently low, you can say with a certain level of confidence that the sample you have was not taken from the population in question.  This is useful when you want to suggest that something other than random chance has led to this observed difference, such as a medical treatment or tutoring.  

The probabilities are gotten by first calculating a test statistic (z-score/statistic for a z-test, t-score/statistic for a t-test), then using either a Cumulative Density Function or Survival Function (depending on which tail you care about) to determine the area under the curve (a normal distribution for z-test, and T distribution for t-test) outside of your test statistic value, giving you the probability of a sample producing a test statistic with a magnitude at least that of your sample.

There are two important things to keep in mind with z and t tests: 

First, a z-score is the same as an effect size (distance from sample mean to population mean in standard deviations) only when you are taking a sample containing a single measurement.  There are two formulas you will see for a z-score:

<center>![z-score 1](https://clavelresearch.files.wordpress.com/2019/03/z-score-population.png?w=300) 
and 
![z-score 2](https://study.com/cimages/multimages/16/zscoreformulaone.png)</center>

The thing to remember is that these two are the SAME formula, it is just that the one on the left is assuming that you are sampling a single element, not a group, in which case n=1, and thus part of the equation is cancelled out.  The version on the right is the one to use for any sample size larger than 1, as it takes into account that the probability of seeing any given size of difference between your sample mean and the population mean will decrease as your sample size increases, since the randomness of each data point measured in your sample will smooth out their individual extremities, and with increasingly large sample sizes, converge on the population mean.  You can think of this intuitively because if your sample size could get large enough, you would eventually be sampling the entire population, and the sample mean would invariably be the same as the population mean no matter how many trials are done.  By dividing the denominator by the square root of n, the equation on the right effectively increases the absolute value of your z-score as the value of n increases, which will reduce the resulting probability (area under curve) calculated from the normal distribution.

The second thing to remember is the rule of thumb for when to use a t-test rather than a z-test, which is when both of these conditions are true:
1. You do not know the standard deviation for the population being tested against
2. You have a sample size under 30

You can think of a t-distribution as a normal distribution with an extra parameter (known as "degrees of freedoom") that make the tails fatter when its value is low.  Degrees of freedom is calculated as n-1 (one less than sample size), so a small sample size leads to fatter tails on the distribution.  The reason for this adjustment is that when you do not know the standard deviation of the population, you are forced to use the one from your sample as an estimate, and these estimates will be less reliable at smaller sample sizes, thus the fatter tails at small sample sizes increase the probability of getting larger mean differences, since your estimate of the spread of the population has introduced some potential error.  As sample sizes get above 30, the point estimates for population standard deviations from a sample get more reliable, and the t-distribution becomes almost identical to the normal distribution.  

The formula for taking the standard deviation of a sample (to use as a point estimate) is:

<center>![Sample Standard Dev.](https://www.gstatic.com/education/formulas/images_long_sheet/sample_standard_deviation.svg)</center>

I encourage you to read about Bessel's Correction in the formula for standard deviation for a sample on your own, which is using (n-1) as a denominator in calculating variance rather than just n.


Let's get back to our flight example:

You take the first reading of 80dB by itself just to get a rough idea of what you're dealing with, to decide if the conversation they are having is indeed loud (the alternative hypothesis), or if it is just really early.  Thus, you want to know what the chances of getting a reading like this would be if it were a normal conversation (the null hypothesis), which you know occur (for the sake of this post) with a mean on 60dB and standard deviation of 10dB.  Since you know the standard deviation of the population you are testing against, a t-test will not be necessary.  Also, since you have only sampled one loudness reading, you know that the effect size (distance of sample mean from population mean) will be the same as the z-score, which you can calculate in your head as (80-60)/(10) = 2.

As a budding statistician, you're aware that a z-score of 2 means the probability outside of 2 standard deviations on a normal distribution, which you remember adds up to about 5% in both tails.  You're only interested in the chance of getting a louder reading, so you divide this in half to find the area under only the upper tail, to roughly 2.5%.  So far, your single data point has showed that there was only a 2.5% chance of getting a loudness reading this high from a normal conversation, which is sufficiently low to satisfy your 95% confidence interval on calling the conversation loud. 

Further testing is in order.  You begin to compile more readings, and notice that with each new reading, your sample mean has been slowly rising, and your sample size increasing, which is raising your z-score with each sample, making your more and more confident that you are indeed dealing with a level of loudness which is statistically significant.  After several rejections of the null hypothesis (that this is a normal conversation), you are confident that your sample has been taken from a loud conversation, but how loud?

Over the course of 25 minutes, you obtain 25 loudness readings sampled from the conversation.  The mean loudness from the sample of the conversation thus far is calculated to be a blaring 90dB, with a sample standard deviation of 10dB, bringing on justifiable fears of hearing loss from exposure.  You Google the issue to determine your health risk, and find that exposure time of 15 minutes to 100dB average loudness is considered dangerous.  You don't want to be rude, but you decide that if there is more than a 1% chance of you sustaining permanent hearing loss in the next 15 minutes, you will have to be candid about your concerns with your neighbor.

To determine this, you structure a hypothesis test that can utilize your data. You decide that if the true mean volume of their conversation is 100dB, then you are at risk of hearing loss, so the null hypothesis is that the conversation is actually at 100dB, with the alternative hypothesis that it is below 100dB, where it should do no long-term harm to your hearing.  You have already reached an internal agreement that anything above a 1% chance of hearing loss is enough to accept the null hypothesis and break the peace, and anything less will be sufficient to prove to yourself that the ringing developing in your ears should only  be a temporary burden (the alternative hypothesis). Thus, you want to find the probability of drawing your sample from a conversation with a mean volume of 100dB, and hope that probability is beneath your alpha value of 0.01.

Now you contemplate the use of either a z or a t test...  You realize that for this test, you do not know the actual standard deviation of their conversation because you haven't taken a measurement for every word they've spoken the entire time, so you'll have to estimate it from your sample of 25 measurements.  Since your sample size is less than 30, and you're using a point estimate for standard deviation, you know that both boxes are checked for using a t-test.  This will be a one-sample t-test, since you are comparing your sample to a theoretical population.  The formula for a t-statistic in a one-sample t-test is identical to the formula for a z-score, save that the population standard deviation is replaced by your point estimate, shown below:

<center>![one-sample t-statistic](https://ariqfazari.files.wordpress.com/2013/06/chapte171.gif?w=292&h=153)</center>

Plugging in the numbers, you get t = (90-100)/(10/sqrt(25)) = -5, which you calculate from a t-distribution with 24 degrees of freedom to have a probability very close to zero.  This means that based on your sample, there is almost a zero percent probability that their conversation is at a volume which will cause any long term hearing loss.  Thus, the null hypothesis of their conversation being at 100dB is rejected in favor of the alternative hypothesis that it is quiet enough not to cause health concerns.  Nevertheless, your short term goals of a nap are important, so you put on your almost cartoonishly sized Smith&Wesson ear protection, and proceed to nod off.

Several hours later, you wake up to find your neighbor is still talking.  With literally nothing better to do, you decide to find out if they are still having the same loud conversation as before (the null hypothesis), or if they have started to fatigue and moved on to a new, significantly quieter conversation (the alternative hypothesis).  You take another sample, but you have limited time, so you only get 10 measurements.  The sample mean is 75dB, and the sample standard deviation is 3dB.  Now you are testing two samples against one another, meaning you will need a two-sample t-test, but since the standard deviations are not the same, you know that you should use a Welch's t-test rather than a student's t-test, which has different formulas for t and degrees of freedom (v), but which conveniently pools your sample variances. Below is the formula for t in a Welch's t-test:

<center>![Welch's t-statistic](https://ncalculators.com/images/formulas/t-test-formula.jpg)</center>


Using this, you calculate your z statistic to be (75-90)/sqrt((100/25)+(9/10)) = -6.78, but you remember that for a Welch's t-test, the calculation for degrees of freedom is not as simple as adding your sample sizes together and subtracting 2, as it is for a student's t-test.  For a Welch's t-test, it's a bit more complex.  The formula is shown below:

<center>![Welch's t-test Degrees of Freedom](https://wikimedia.org/api/rest_v1/media/math/render/svg/2108692a7e5ce58c5bbbc3a34720411b64a1922e)</center>

This calculates to 31.73 (if variances had been equal, a student's t-test could have been used, and the degrees of freedom would have been 25+10-2 = 33, so we are in the same ballpark).  Passing these values for t and v into the CDF for a T distribution, you find another near zero probability that the samples are taken from the same conversation. Thus, the null hypothesis that they are still having the same loud conversation is rejected, and you conclude that your neighbor has indeed piped down.


