# HMM-Library
A Hidden Markov Model library in Python (+NumPy)

This dates from a few years back (2011) but I haven't seen anything like it after looking around, so I've decided to publish it.

I need to document it properly but other than that, it's quite functional and performs really well.

Some reorg of the folders is necessary. I have only used this myself for research.

I've included gprof2dot.py by José Fonseca, which isn't strictly required by I used it to plot the output of the profiler.

The heavy-lifting happens in hmm.py

The file precisionParameters.py contains constants related to the algorithms. They should be self-explanatory if you know the algorithms used.

These two files alone are all you need to run the library. Could easily be just one file by simply including the constants in hmm.py, however I wanted to keep track of changes in the algorithms proper clearly separate from changes in mere constants.

The file support.py includes generic functions used in the tests, like timers, sequence generators, etc.
