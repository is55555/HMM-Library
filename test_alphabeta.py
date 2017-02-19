from random import shuffle
from support import timecall
from hmm import HMM
from numpy import take

def time_alphabeta(size_states, size_observations, op="alpha"): # by size here I mean the size of the space (hidden state space and observation space AKA "alphabet")
    S = range(size_states)
    O = range(size_observations)
    OBS = range(size_observations)*4
    shuffle(OBS)
    test = HMM( S, O )
    test.random_initialisation()

    print OBS
    print test

    test1 = HMM( S, O, test.A, test.B, test.pi )
    bo = take( test.B, OBS, 0 )
    print "bo = take( test.B, OBS, 0 )"
    print str(bo)
    print "A.shape  = ", test.A.shape
    print "bo.shape = ", bo.shape
    a1, s1 = test1.alpha_scaled( test.A, bo, test.pi )
    print "a1,s1",a1,s1
    print "s1.shape = ", s1.shape
    if op == "alpha": timecall( "alpha scaled  ", test1.alpha_scaled, (test.A, bo, test.pi) )
    else:             timecall( "beta scaled   ", test1.beta_scaled,  (test.A, bo, s1) )


def mainrun():
    print " ***** alpha ***** "
    time_alphabeta(10, 15, "alpha")
    print " ***** beta ***** "
    time_alphabeta(10, 15, "beta")

#mainrun()

import profile
profile.run('mainrun()', 'test_alphabeta.profile')
