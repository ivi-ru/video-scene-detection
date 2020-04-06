# Video Scene Detection based on Optimal Sequential Grouping algorithm

Here you can find implementations of algorithms from the following papers:

- Robust and Efficient Video Scene Detection Using Optimal Sequential Grouping [https://ieeexplore.ieee.org/abstract/document/7823628]
- Optimally Grouped Deep Features Using Normalized Cost for Video Scene Detection [https://dl.acm.org/doi/10.1145/3206025.3206055]

The task is temporally dividing a video into semantic scenes.

Both papers propose the following steps to achieve the goal:
- divide a video into shots (sequence of frames from one editing cut to another)
- extract an arbitrary set of features from each shot
- find pairwise distances between feature vectors
- find such sequence of squares along main diagonal that will be optimal from the point of some cost function

These squares will represent the desired scenes.

The core idea is to solve the problem as an optimization process of sequential grouping task.

The first paper describes optimization of the metric caller H_add – simple sum of distances inside squares.
The second paper is about H_nrm - improvement of the H_add that use normalization.

This repo doesn't contain feature extraction and pairwise distances calculation steps.

Its goal is to demonstrate algorithms implementations on some synthetic data.
