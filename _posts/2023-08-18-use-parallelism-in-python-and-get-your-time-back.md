---
layout: post
title: "Use Parallelism in Python, and Get Your Time Back"
summary: "If you’re programming in Python and not using parallelism where possible, chances are you’re not getting as much done as you could be."
author: natecibik
date: '2023-08-18 21:27:28 +0000'
category: Python
thumbnail: /assets/img/posts/use-parallelism-in-python-and-get-your-time-back-01.jpeg
keywords: python, parallelism, multiprocessing, concurrency, performance
permalink: /blog/use-parallelism-in-python-and-get-your-time-back/
usemathjax: true
---

> *This post was originally published on [Medium](https://medium.com/@natecibik/use-parallelism-in-python-and-get-your-time-back-987fb10e5bc) on August 18, 2023. It has been migrated to hiddenlayers.tech as the canonical version.*

If you’re programming in Python and not using parallelism where possible, chances are you’re not getting as much done as you could be. With parallelism, we can abbreviate the time it takes to get answers, giving us more time to go outside, walk the dog, and smell the flowers. True to form, Python provides a simple and intuitive interface for setting up concurrent workflows. However, incorporating them into our work can be intimidating, because improper use of these tools can lead to unexpected errors in our output or bugs that are difficult to track down, and in some cases we can actually *slow down* our code drastically. In this article, we will learn to protect our precious time using the `concurrent` python library, best practices, and some pitfalls to avoid. We will cover both the `ThreadPoolExecutor` and the `ProcessPoolExecutor`, and tips on how to know when to use one or the other.

#### The Woodscape Dataset

For an example use case, we will look at exploring the [Woodscape autonomous vehicle dataset from Valeo](https://woodscape.valeo.com/woodscape/dataset), which was collected through four fisheye cameras capturing a surround view from three different vehicles in European locations. The relative small size and simple file schema of this dataset makes it ideal for this exploration.

![Woodscape example](/assets/img/posts/use-parallelism-in-python-and-get-your-time-back-02.jpeg)

#### Dataset Statistics

Whenever we are engaging with a machine learning dataset, it is always best to first investigate its idiosyncrasies through some statistical analysis. This is because all datasets will be different, that is, they will be biased to the operational domain that they were collected in. Understanding these biases will inform us on where the weaknesses of our model training might be, and gives us insight on the adaptations we should make to the training procedure in order to account for them. For instance, [the performance of object detection networks can suffer from imbalance problems](https://arxiv.org/pdf/1909.00169.pdf) in the input data, and based on the nature of these imbalances, various sampling strategies can be employed to ameliorate their negative effects. Similarly, [semantic segmentation network performance will suffer if class imbalances in the data are not taken into account](http://cs.uccs.edu/~jkalita/work/reu/REU2017/16Small.pdf).

#### A Brief Intro to Concurrency

So, it is clear that we need to evaluate the statistics of our dataset to get the best training results. But before you set up that massive for loop iterating over all of the training data, consider to yourself whether the iterations of that loop need to share information or have any effect on one another, because this is the key question in deploying parallelism. In this case, we want to iterate over each training frame of a dataset and gather some information about each one. A simple approach might be to create a dictionary, pandas DataFrame, or some other data structure and populate it during iteration, then compute statistics over the final result. This is certainly what we need to do, but the fact is we actually don’t need to combine the information from the individual frames until we’re done gathering it all, which means that we can parallelize the frame parsing.

To understand why this might be helpful, let’s talk a bit about multiprocessing and multithreading.

- **Multiprocessing** is truly parallelized. By default, the number of processes is set to the number of logical cores in your CPU, and the tasks are completed in isolation and without interruption in their respective cores. [The number of processes can be tuned based on your use case](https://superfastpython.com/multiprocessing-pool-num-workers/), for instance, if your tasks are particularly computationally intense, you might consider setting the number of processes to the number of physical rather than logical CPU cores. For less intensive tasks that benefit from parallelism like IO, you may set the number of processes higher than the number of logical cores so that each core is responsible for more than one task, which it will cycle between (context switching). However, if you’re thinking about assigning many more than one task per core, it’s probably better to use **Multithreading**.
- **Multithreading** is not actually parallel. In fact, only one thread runs at any given time, but your CPU rapidly switches between these threads, doing a bit of work on each one as it goes around. It’s natural to wonder: how does this lead to faster performance? Shouldn’t all of that context switching in between doing work actually slow the whole process down? This is a fair question, because the answer is actually yes when the only work getting done is by your Python program. An example can help with the confusion here. Imagine that you need to write your name on 10 sheets of paper arranged in a circle around you. If your name is 5 letters long, and you take a multithreaded approach by switching to the next sheet of paper each time you write a single letter, you’re going to context switch 50 times, wasting time since it would be easier to write all 5 letters on each page and only reposition yourself 10 times instead. The multithreaded approach is not helpful in this example because you’re the only one responsible for doing any work at each station. Imagine instead that you’re in a timed sushi eating contest, surrounded by 10 chefs each preparing a total of 5 pieces of sushi for you, but they can only make you one at a time. If you eat a piece of sushi from one chef, it will save time to go eat the pieces which are ready at the other stations while you wait for the them to prepare their next piece. Now the rapid context switching is working to your advantage, because you are not wasting time waiting at each station for someone else to do their job, and you’re unblocking the other chefs in the process. To take this example back into the computer world, if we’re waiting on one thread to load the contents of a file, we’re better off going and getting the results from other threads while we wait.

Now the reason for our interest becomes clear: there’s no reason to sit and wait for each file to load separately, we can employ the help of many workers to each do their individual task, and then we can focus on collecting the results once they are ready, the same way we won the sushi eating contest. But enough metaphor, let’s look at this in practice by exploring the Woodscape dataset.

#### Load a frame and inspect

Let’s load and visualize the first frame in the dataset to get our first look at the data. In this tutorial we assume that the Woodscape dataset is downloaded locally at `./woodscape/`. The semantic masks are stored as `.png` files, with numeric names matching their corresponding RGB frames.

```
['00000_FV.png', '00001_FV.png', '00002_FV.png', '00003_FV.png', '00004_FV.png']  
['00000_FV.png', '00001_FV.png', '00002_FV.png', '00003_FV.png', '00004_FV.png']
```

#### Overlay semantic mask to test data loading

To make sure we’ve got our data lining up, let’s make a quick overlay of an RGB and semantic mask

![jpeg](/assets/img/posts/use-parallelism-in-python-and-get-your-time-back-03.jpeg)

#### Exploring Statistics

Now that we understand our data schema, we can accumulate some statistics.

#### Class Pixel Counts

The most foundational question we can ask about semantic segmentation training data is how the class pixel distribution looks, that is, the relative number of pixels (examples) for each class in the dataset. This will instruct how we should architect the training process to get the best performance from our model. Let’s investigate whether we have semantic class imbalances.

#### Define our scraping function

Here we’ll define a function that is parallelizable, which means that it does not need to reference any object that is used by other processes. It is generally dangerous to mutate an object in the global scope from multiple threads at once, so it is best practice to have the inputs and returns be independent of what happens in any other thread, then compile the results back in the main thread. Below we have a simple function to return the total pixel count per class for a given frame of data, then we can compare the performance of using a for loop to using multithreading.

Test scraping function on a single frame

```
{'void': 877494,  
 'road': 310862,  
 'lanemarks': 1489,  
 'curb': 5641,  
 'person': 13124,  
 'rider': 0,  
 'vehicles': 27608,  
 'bicycle': 0,  
 'motorcycle': 0,  
 'traffic_sign': 262}
```

#### Doing things the slow way

In order to make the point, we’ll need to compare how slow it is to compile this information using a for loop to iterate over all frames.

```
start time: 1692347509.286457  
end time: 1692347770.425536  
Gathering data took 261.1390788555145 seconds. (4.35 minutes)
```

Gathering this information with a for loop took ~260 seconds, that’s nearly 4 and a half minutes. Imagine if our dataset was much larger…

#### Bring in the `ThreadPoolExecutor`

Now, let’s break out the `ThreadPoolExecutor` and see if we can improve this. We can iteratively make submissions to the thread pool which will give us a set of futures, which are promises of results that haven't come back yet. Using the `as_completed` function allows us to grab these results as soon as they're available without having to wait for the others, just like eating whichever piece of sushi is ready while you wait for the others to be prepared.

```
start time: 1692347770.444041  
end time: 1692347826.534175  
Gathering data took 56.09013390541077 seconds. (0.93 minutes)
```

This is significantly faster! We’re at less than 1/4 of the time it took to run the for loop.

#### Running the same test with `ProcessPoolExecutor`

The `ProcessPoolExecutor` employs processes instead of threads, so the number of these processes is limited by the number of CPU cores. However, it can be much faster for very computationally heavy tasks. Our task is running summations over semantic masks, which is sort of in the middle on the computational scale, so it isn't immediately obvious which would be faster, and we should try both.

To use the `ProcessPoolExecutor` in a Jupyter notebook, we’ll have to load the function from a python file. We can export one easily with the following cell.

```
start time: 1692347875.697177  
end time: 1692347941.897485  
Gathering data took 66.20030808448792 seconds. (1.1033384680747986 minutes)
```

We can see that using true parallelization with the `ProcessPoolExecutor` has led to slightly slower results, so we can conclude our function is sufficiently lightweight for using threads rather than processes.

#### Observing class distribution

Now that we’ve got our pixel counts, we can see if we have class imbalances in our data.

![png](/assets/img/posts/use-parallelism-in-python-and-get-your-time-back-04.png)![png](/assets/img/posts/use-parallelism-in-python-and-get-your-time-back-05.png)

We can see that some classes are extremely under-represented compared to others. Semantic segmentation models have been observed to have the same issue as image classifiers when trained on randomly sampled, unbalanced data, which is a tendency to overclassify the majority classes because errors on these classes create the heaviest and most influential loss during training. It is very likely that a semantic segmentation network trained on this data would benefit from a class balancing sampling strategy.

#### Classwise Heat Maps

Now let’s try for something heavier, and see how our two choices from the `concurrent.futures` package compare. For this example, we'll accumulate classwise heatmaps by expanding the semseg masks into the same number of dimensions as there are classes, where each mask then represents a binary mask for that class. Doing this will allow us to sum all of these larger arrays together to get classwise heatmaps showing us where the classes appear the most frequently.

#### Define our mask scraping function

Let’s make sure this function is working as we expect by testing it on a single frame. We should see the road highlighted.

![png](/assets/img/posts/use-parallelism-in-python-and-get-your-time-back-06.png)

#### Doing it the slow way

Let’s run a for loop to get a baseline of how long this takes running in series.

```
start time: 1692348388.025281  
end time: 1692349306.348945  
Gathering data took 918.3236639499664 seconds. (15.31 minutes)
```

Ouch, this time our analysis takes about 15 minutes. This would not scale well if our dataset was large. We’re going to need to use some form of concurrency.

#### Inspecting a heatmap

Let’s take a look at the heatmap for the “person” and “vehicles” classes.

![png](/assets/img/posts/use-parallelism-in-python-and-get-your-time-back-07.png)

#### Trying the `ThreadPoolExecutor`

Running the following cell will actually crash our notebook because we hit RAM limitations. The reason for this is that we have 2 large arrays being held in memory in a large number of threads before they can finish, and so the lack of metering in our memory sets us up for failure.

```
Cannot execute code, session has been disposed. Please try restarting the Kernel.  
  
  
  
The Kernel crashed while executing code in the the current cell or a previous cell. Please review the code in the cell(s) to identify a possible cause of the failure. Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.
```

#### Multithreading IO only

Let’s see what happens if we move the parsing of the binary masks from the multithreaded function into the main thread, and only do image loading in the threads.

```
start time: 1692349727.674081  
end time: 1692350332.025268  
Gathering data took 604.3511869907379 seconds. (10.07 minutes)
```

This adjustment has kept us from going OOM, and we’re down to 10 minutes to run the full dataset. This 33% improvement over the for loop shows us that just multithreading the IO can speed things up considerably, but we can do even better. Let’s explore going back to threading the heavy function, but this time we’re going to create a queue to meter our submissions to the threadpool.

#### Using Queued Thread Pool Submission

```
start time: 1692351512.391178  
end time: 1692351843.313716  
Gathering data took 330.9225380420685 seconds. (5.52 minutes)
```

Look at that! We’re down to 5 and a half minutes, that’s 55% of the time it took to run the IO only test, and only 33% of the time to run the original for loop.

#### Testing with the `ProcessPoolExecutor`

Let’s see what a queueing approach using the `ProcessPoolExecutor` can offer us timewise. Again, we need to write out a python file with the function we'd like to parallelize.

To keep our CPU from locking up, let’s set the max\_workers to 1 less than the CPU count. Note that we are using the same queue size as above for comparison, but with process pools it may be more effective to use smaller queues. During my tests of this example, I found that the difference was negligible, so I’m keeping it 100. Feel free to experiment for yourself!

```
start time: 1692351843.368718  
end time: 1692352992.659966  
Gathering data took 1149.2912480831146 seconds. (19.15 minutes)
```

Oh, dear. We’re actually worse off in terms of time than we were with a simple for loop, with much more code complexity, and our CPU was far more bogged down while the job ran. Clearly the multiprocessing is not the way to go here, and the `ThreadPoolExecutor` is the overall winner!

#### Conclusion

For the lightweight function returning pixel counts, a simple use of the `ThreadPoolExecutor` gave us a 4x faster speed compared to a plain for loop. For the heavier heatmap function, it was necessary to create a submission queue, but the `ThreadPoolExecutor` was there to save our youth once more with a 3x speedup over a for loop. With all of this newfound time, I think I'll go for a walk.

While the `ThreadPoolExecutor` is clearly the winner in our investigation, in other contexts the `ProcessPoolExecutor` will be the more appropriate choice for task concurrency, so it's important to test our options to find the best one for our use case.

![png](/assets/img/posts/use-parallelism-in-python-and-get-your-time-back-08.png)

#### Closing Thoughts

In this investigation, we’ve seen that concurrency can drastically speed up our workflows. However, if we’re not careful, it can also cause a variety of problems, including the OOM issues and slowdowns we’ve seen here. It’s outside of the scope of this article to demonstrate, but users should be very careful about any global objects that are being interacted with by threads or processes when using concurrency, and keep the activity in their threads completely independent whenever possible. If we keep these simple rules in mind, then we can confidently deploy concurrency in our workflows, and have more of our lifes left to spend eating sushi!

![Enjoy sushi](/assets/img/posts/use-parallelism-in-python-and-get-your-time-back-09.jpeg)
