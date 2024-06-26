# FinalProject

This folder contains the code for my CMP 3rd year final project "A Survey, Analysis And Evaluation Of
Machine Learning Models For
Recommendation Systems". The folder contains all the code, data and visualisations that was used
in my report and includes extras that were omitted from the study
for various reasons.

The folder names should be inutive but for full disclosure 
`own_algorithms` contains my the hybrid algorithms built using `Surprise.algobase`
and various helper function including `top-n` and `testing_algorithm`.

Folders `KNN Graphs` and `Matrix Graphs` contain the graphs for
each independent analysis. `data_graphs` contains graphs that was used in methodology for data exploration
and `predictions` contain csvs for user 1, 134, and 398 with predictions for the algorithms used.

It should be noted the notebooks are ordered, and this is the order they should be run in, if this required.
Failing to do so may cause odd results as each notebook reads data created from a previous notebook. This excludes the data preface 
notebooks

I should also be noted, some algorithms trialed in this study could cause heavy
strain on CPU load.