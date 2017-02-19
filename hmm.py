# -*- coding: utf-8 -*-

import cPickle
from numpy.random import random, random_integers
from numpy import e, array, ones, zeros, add, searchsorted, \
     product, dot, multiply, alltrue, log, equal, newaxis, \
     take, empty_like, allclose, where

from precisionParameters import *

from copy import deepcopy

def random_matrix(shape, dimNorm = 0):
    """ the matrix is uniform and normalised to add up to 1 in the specified dimension """
    matrix = random(shape) # this is numpy's random, creates arrays of random numbers
    # ... when passed an integer, and a multidimensional array when passed a tuple
    # ( v.g. random((3,2)) would return a random 3X2 array (uniforms in each cell))
    return normalise(matrix, dimNorm)
    
def normalise(matrix, dimNorm):
    if dimNorm: matrix /= add.reduce(matrix, dimNorm)[:, newaxis] 
    else: matrix /= add.reduce(matrix)
    # now we have a matrix of random uniforms normalised to sum "1" on the...
    # ...specified dimension. Like array([ 0.3291846 ,  0.37059381,  0.30022159]) or like:
    #array([[ 0.46175359,  0.53824641],     # <- these add up to 1
    #       [ 0.78660667,  0.21339333],     # <- these add up to 1
    #       [ 0.32146702,  0.67853298]])    # <- these add up to 1
    # this last one was normalised over dim 1 (second dim)
    return matrix # does it in place anyway, so this output can be ignored if we have the...
    #... original matrix in the current scope of the call

def is_a_prob_matrix(matrix, dimNorm):
    """checks if 'matrix' is a proper probability matrix normalised over the dimension dimNorm"""
    reduced_matrix = add.reduce(matrix, dimNorm) - 1
    return ( alltrue(reduced_matrix < EPSILON) and alltrue(reduced_matrix > -EPSILON) and 
                 #   alltrue(alltrue(self.A<=1.0)) && alltrue(alltrue(self.A>=0.0)) )
                 alltrue(matrix <= 1.0) and alltrue(matrix >= 0.0) )
    
    # ******************** specific auxiliary functions:
def _bwm_correct_matrix( M, axis, p, verbose = False ): # _bwm => Baum-Welch for multiple observations - this function is called profusely there
    """Avoids divide by zero errors.
    Doesn't change the result of the algorithm.

    Note that "normalise" wouldn't work here because there would be a divide by 0
    error. Anyway the idea is the same. Normalise could also be extended to take care
    of these cases but I didn't want to integrate it so much, just in case this ended
    up being called often in iterations (optimisation reasons).

    You can only have 0 probabilities if your observation matrix contains symbols
    that don't appear in your observations AND the initial state transition and
    observation probabilities are such that a state is reachable only if you observe
    those symbols.
    Parameters are:
    M    : matrix
    axis : axis along which we need a pdf
    p    : value to replace with (usually 1/M.shape[k]) - otherwise the matrix 
           wouldn't be correct. Passing the value to avoid a call to 
           numpy.shape (might be insignificant though)
    """
    mask = equal(add.reduce(M, axis), 0.0)
    if axis == 1:
        for i in xrange(M.shape[0]):
            if mask[i]: M[i, :] = p
    elif axis == 0:
        for i in xrange(M.shape[1]):
            if mask[i]: M[:, i] = p
    else: raise "implemented only for axis 0 and axis 1"

    if verbose: print "corrected", axis, p,
    return M

# ----------------------------------------------------------------------

class HMM:
    def __init__(self, state_list, observation_list,
                 A = None, B = None, PI = None, check = False):
        self.N = len(state_list)
        self.M = len(observation_list)
        self.STATES = state_list
        self.OBSERVATIONS = observation_list
        # if all A, B and PI are unset, we initialise to an
        # equiprobable HMM, which is probably the simplest one.
        # (see method "equiprobable_initialisation")
        if A is None:  A  = ones((self.N, self.N), float) / self.N
        if B is None:  B  = ones((self.M, self.N), float) / self.M
        if PI is None: PI = ones((self.N,       ), float) / self.N

        self.A = array(A, float)
        self.B = array(B, float)
        self.pi = array(PI, float)
        self.index_HMM()
        if check: self.integrity_check()
	else: self._cap(norm = True)

    def A_B_PI(self, copy = False):
        A, B, PI = self.A, self.B, self.pi
        if copy: return deepcopy(A), deepcopy(B), deepcopy(PI)
        else: return A, B, PI

    def random_initialisation(self, check = True):
        """sets A, B and PI to random values, 
        but preserving stochastic properties."""
        self.A  = random_matrix(self.A.shape, dimNorm = 1) # over second dim
        self.B  = random_matrix(self.B.shape)
        self.pi = random_matrix(self.pi.shape)
        if check: self.integrity_check()

    def equiprobable_initialisation(self, check = True):
        """sets A, B and PI to be completely equiprobable.
        All transitions have the same probability, and so do all observations
        and also all initial states are equally probable.
        
        The default HMM is this at creation time, unless probability
        matrices are specified."""
        A = ones( (self.N, self.N), float) / self.N
        B = ones( (self.M, self.N), float) / self.M
        PI = ones( (self.N,), float ) / self.N
        self.A  = array(A,  float)
        self.B  = array(B,  float)
        self.pi = array(PI, float)
        if check: self.integrity_check()

    def make_ergodic(self, check = True): 
        "make all transitions possible "
        self.A = self.A + equal( self.A, 0.0 ) * EPSILON # should maybe use "where < EPSILON" instead
        normalise(self.A , dimNorm = 1)
        if check: self.integrity_check()


    def make_nothing_impossible(self, check = True): 
        "make all probabilities (A, B, PI) POSSIBLE"
        self.A  = self.A  + equal( self.A , 0.0 ) * EPSILON # should maybe use "where < EPSILON" instead
        self.B  = self.B  + equal( self.B , 0.0 ) * EPSILON
        self.pi = self.pi + equal( self.pi, 0.0 ) * EPSILON
        normalise(self.A , dimNorm = 1)
        normalise(self.B , dimNorm = 0)
        normalise(self.pi, dimNorm = 0)
        if check: self.integrity_check() 

    def normalise_hmm_by_reachability(self, blind = False):
        """ reorder the states so that the most reachable come first 
        in the matrices.
        Note that reachability ponders PI too.
        This works best when average chain length is close to N (by design). It should be
        changed if this is not the case. If chains are much longer than N, call it
        with ignorePI set to True.

        Works quite well for model comparison."""
        if blind:    # when training a HMM unsupervised the names of states and observations
                # are arbitrary, and thus so are their order. To compare them meaningfully 
                # one needs a normal form. Here we take the STATES and OBSERVATIONS as defined
                # and restore them in the end after the normalisation. The HMM learn would only
                # capture the dynamics but has no way to know what's called what.
            origSTATES       = self.STATES[:]
            origOBSERVATIONS = self.OBSERVATIONS[:]
        reach = add(add.reduce(self.A,0), self.pi)
        # permutation = reach.argsort()[::-1] # this causes a problem with "ties"
	permutation = reach[::-1].argsort()
        ret = self.reorder(permutation)
        if blind:
            self.STATES = origSTATES
            self.OBSERVATIONS = origOBSERVATIONS
            return ret
        else: return ret


    def normalise_hmm_by_A(self):
        """ reorder the states so that the transition probability 
        meets A[0,0] >= A[1,1] ... >= A[n,n] """
        permutation = self.A.diagonal().argsort()
        permutation = permutation[::-1]
        return self.reorder(permutation)

# normalise_hmm_by_PI is also an option to consider...

    def reorder_to_match(self, list_of_states):
        assert len(list_of_states) == len(self.STATES)
        permutation = []
        for i in list_of_states:
            permutation.append(self.state_index[i])
        self.reorder(permutation)

    def reorder(self, permutation):
        """ reorders the HMM according to a permutation

        permutation must be array-like, and maps a state index
        into a new state index.

        V.g. permutation[2,1,3,0] would send the state 
        at position 0 to position 2, the state at position 1 to position 1
        (same), the state at position 2 to position 3 and the state at 
        position 3 to position 0. All the indices would remain consistent
        to these changes in all matrices. """
        STATES = []
        A  = empty_like(self.A)
        B  = empty_like(self.B)
        PI = empty_like(self.pi)
        N  = A.shape[0]
        for i in xrange(N):
            for j in xrange(N): A[i, j] = self.A[permutation[i] , permutation[j]]
            B[:, i] = self.B[:, permutation[i] ]
            PI[i] = self.pi[permutation[i]]
            STATES.append(self.STATES[permutation[i]])

        self.STATES, self.A, self.B, self.pi = STATES, A, B, PI # do it in-place
        self.index_HMM() # re-index
        return A, B, PI

    def __str__(self):
        " string representation of the HMM "
        ret  = "=" * 80
        ret += "\nStates: "+str(self.STATES)
        ret += "\nN : " + str(self.N) + " (size of States vector)"
        ret += "\nObservations: "+str(self.OBSERVATIONS)
        ret += "\nM : " + str(self.M) + " (size of Observations vector)"
        ret += "\n" + ("-" * 80)
        ret += "\nState transition probabilities (A vector):"
        ret += "\n"
        for i in xrange(self.N):
            for j in xrange(self.N):
                x = self.A[i][j]
                if x < EPSILON: x = 0
                ret += " %7E " % x
            ret += "\n"
        ret += "\n" + ("-" * 80)
        ret += "\nObservation probabilities (B vector):"
        ret += "\n"
        for i in xrange(self.M):
            for j in xrange(self.N):
                x = self.B[i][j]
                if x < EPSILON: x = 0
                ret += " %7E " % x
            ret += "\n"
        ret += "\n" + ("-" * 80)
        ret += "\nInitial probabilities (Pi vector):"
        ret += "\n"
        for i in xrange(self.N):
            x = self.pi[i]
            if x < EPSILON: x = 0
            ret += " %7E " % x
        ret += "\n" + ("*" * 80) + "\n"
        return ret

    def __repr__(self):
        " reproducible string representation of the HMM "
        ret  = "HMM("+str(self.STATES)+", "+str(self.OBSERVATIONS)+",\n"
        ret += repr(self.A)+",\n"+repr(self.B)+",\n"+repr(self.pi)+")"
        return ret

    def __str__2(self):
        " string representation of the HMM "
        ret  = "=" * 80
        ret += "\nStates: "+str(self.STATES)
        ret += "\nN : " + str(self.N) + " (size of States vector)"
        ret += "\nObservations: "+str(self.OBSERVATIONS)
        ret += "\nM : " + str(self.M) + " (size of Observations vector)"
        ret += "\n" + ("-" * 80)
        ret += "\nState transition probabilities (A vector):"
        ret += "\n" + str(self.A) + "\n" + ("-" * 80)
        ret += "\nObservation probabilities (B vector):"
        ret += "\n" + str(self.B) + "\n" + ("-" * 80)
        ret += "\nInitial probabilities (Pi vector):"
        ret += "\n" + str(self.pi) + "\n" + ("*" * 80) + "\n"
        return ret

    def index_HMM(self):
        """Create indices.
        self.state_index       => maps a state to its index
        self.observation_index => maps an observation to its index"""
        self.state_index = {}
        self.observation_index = {}
        for i in xrange(self.N):
            self.state_index[self.STATES[i]] = i
        for i in xrange(self.M):
            self.observation_index[self.OBSERVATIONS[i]] = i
            
    def saveHMM(self, f, saveState = None):
        "dump to disk"
        version = "1.0"
        cPickle.dump( version, f, 1 )
        cPickle.dump( saveState, f, 1 )
        if saveState:
            cPickle.dump( self.STATES, f, 1 )
            cPickle.dump( self.OBSERVATIONS, f, 1 )
        cPickle.dump( self.N, f, 1 )
        cPickle.dump( self.M, f, 1 )
        cPickle.dump( self.A, f, 1 )
        cPickle.dump( self.pi, f, 1 )
        for i in xrange(self.M):
            cPickle.dump( self.B[i, :], f, 1 )

    def __getinitargs__(self):
        """helper for cPickle"""
        return self.STATES, self.OBSERVATIONS, self.A, self.B, self.pi

    def loadHMM(self, f):
        "loads disk dumps made with saveHMM"
        version = cPickle.load(f)
        if version == "1.0":
            saveState = cPickle.load(f)
            if saveState:
                self.STATES = cPickle.load(f)
                self.OBSERVATIONS = cPickle.load(f)
            self.N = cPickle.load(f)
            self.M = cPickle.load(f)
            if saveState:
                self.index_HMM()
            self.A = cPickle.load(f)
            self.pi = cPickle.load(f)
            self.B = zeros((self.M, self.N), float)
            for i in xrange(self.M):
                self.B[i, :] = cPickle.load(f)
        else:
            raise Exception("File format not recognized")

    def integrity_check(self):
        """checks for consistency. See Rabiner for notation and 
        meaning of A,B,N,M,PI """
        assert self.A.shape == (self.N, self.N), "A has to be N*N"
        assert self.B.shape == (self.M, self.N), "B has to be M*N"
        assert self.pi.shape == (self.N, ), " PI has to be a vector of size N"
        assert is_a_prob_matrix(self.A, dimNorm = 1), \
                "A must be a probability matrix : "+str(self.A)
        assert is_a_prob_matrix(self.B, dimNorm = 0), \
            "columns of B must be a probability matrices (vectors, 1-dim each) : "+str(self.B)
        ##if len(self.pi)==0: return # a zero length vector is reduced to a scalar and makes the following test fail
        assert is_a_prob_matrix(self.pi, dimNorm = 0), "PI must be a probability matrix (vector) " + str(self.pi)

    def set_A(self, s1, s2, value):
        """set A[s1,s2] to value
        (the probability of a transition from state s1 to state s2)
        s1 and s2 expressed as their state representation.
        To set it directly by index no method would be needed:
           A[i1,i2] = value 
           would just work. """
        self.A[self.state_index[s1], self.state_index[s2]] = value

    def set_B(self, s, o, value):
        """set B[o,s] to "value" 
        (the probability of observation 'o' to happen being in state 's')
        both 's' and 'o' and expressed as their representation.
        To set it directly by index no method would be needed:
           B[i_o, i_s] = value 
           would just work. """
        self.B[self.observation_index[o], self.state_index[s]] = value

    def set_PI(self, s, value):
        """set PI[s] = value
        (the probability of being initially in state 's')
        When working directly with indices, just do PI[i_s] = value """
        self.pi[self.state_index[s]] = value

    def get_A(self, s1, s2):
        "(s1,s2) => self.A[self.state_index[s1], self.state_index[s2]]"         
        return self.A[self.state_index[s1], self.state_index[s2]]

    def get_B(self, s, o):
        "(s, o) => self.B[self.observation_index[o], self.state_index[s]]"
        return self.B[self.observation_index[o], self.state_index[s]]

    def get_PI(self, s):
        "(s) => self.pi[self.state_index[s]]"
        return self.pi[self.state_index[s]]

    def _get_observation_indices(self, observations):
        """Get observation indices from their representation.
        
        For example, if we have observation symbol list ['a','b','c','d']
        _get_observation_indices['b','a','d'] will return the array
        array([3,1,0])

        Arrays are internally fixed size, which is why we create a
        zero-array of the same size as the given list and proceed
        to clobber it. A slightly slower but maybe clearer way would
        be to create a list, append to it and then create the array
        from it and return that."""
        indices = zeros( len(observations), int )
        i = 0
        for o in observations:
            indices[i] = self.observation_index[o]
            i += 1
        return indices
    
    def simulate(self, length, show_hidden = False):
        """generates a sequence of observations of given length. This
        sequence will be random and use the probabilities from the HMM.
        if show_hidden is set to True, it will return a list of tuples
        ('s', 'o') with the internal state and the observation generated,
        otherwise it will return a simple list of observations."""
        cumulative_A = add.accumulate( self.A, 1 )
        cumulative_B = add.accumulate( self.B, 0 )
        state = searchsorted( add.accumulate(self.pi, 0), random()) 
            # places the "throw of the dice" in it's slot
        # we have a cumulative distribution of the initial state vector PI
        # and with searchsorted we put this particular sample in its place
        # we put the result in "state" - his same procedure will be used for
        # transitions and observations, following the logical order.
        sequence = []
        states = []
        
        for i in xrange(length):
            states.append(state)
            symbol = self.OBSERVATIONS[ searchsorted( cumulative_B[:, state], random() ) ]
            if show_hidden: sequence.append( (self.STATES[state], symbol) )
            else:           sequence.append(symbol)
            state = searchsorted( cumulative_A[state, :], random() )
        return sequence

    def random_walk(self, length):
        """Random sequence of hidden states (ignoring the probability matrices)
        If you need a random_walk that does consider the probability matrices, that's
        a simulation (see method 'simulate')"""
        rw_i = random_integers(0, self.N - 1, length)
        rw_s = []
        for i in rw_i: rw_s.append(self.STATES[i])
        return rw_s


    def viterbi(self, observations):
        """Viterbi algorithm - most probable complete sequence of internal states
        to match observed sequence. Follows [Rabiner] naming and notation.  """
        A = self.A
        B = self.B
        N = self.N
        T = len(observations)
        O = self._get_observation_indices(observations) # vector of ...
             # ... observation indices rather than their representations.

        # -- initialisation
        ##delta = zeros( N, float )
        delta = multiply(B[O[0]], self.pi) # element-wise multiplication (init) (delta_1 in Rabiner)
        delta_t = zeros( N, float )   # delta_t holds the best probability
        temp = zeros( N, float ) 
        psi = zeros( (T, N), int )    # psi holds pointers to the index of the best probability

        # -- recursion
        for t in xrange(1, T):
            for j in range(N):
                multiply( delta, A[:, j], temp ) # A[:,j] => column at j, as a vector
                psi[t, j] = temp.argmax()      #  PSI_t_j is stored as we traverse the trellis
                       # it stores the most likely previous step leading to j at step (time) t
                delta_t[j] = temp[psi[t, j]] * B[O[t], j]  # delta_t holds the actual probability
            delta, delta_t = delta_t, delta # move on to the next step. In the inner loop delta is really delta_t-1 ...
            #delta = delta_t  # (** this breaks it - I want to look back into this at some point because it's a sneaky bug when it happens. However the line above works.)
            #... and delta_t is created element by element. Now, when we are done with that, delta is ...
            #... updated to be the next one. Note that we are clobbering it every time because the ...
            #... actual result is stored in PSI (the indices for the sequence). The actual probability ...
            #... of every best possible step is not stored.

        # -- termination
        index_star = [delta.argmax()]  # we store here the indices from PSI starting from the best last step
                   # ... which is delta.argmax()
        for psi_t in psi[-1:0:-1]: # psi[-1:0:-1] => take PSI without the first element and reverse it
            index_star.append( psi_t[index_star[-1]] ) 
        index_star.reverse() # we traversed PSI backwards, so we reverse the list to have it in proper order.

        sequence = [self.STATES[i] for i in index_star]   # we return the sequence of states, not their indices
        return sequence
        
    def viterbi_log(self, observations):
        """same, but using a log substitution to avoid precision problems
        
        trick:
        equal( V, 0.0 ) * SMALLESTFLOAT will return a vector such that those ...
        ... positions in V containing a non-zero will contain a zero, and ...
        ... those containing a zero will contain SMALLESTFLOAT.
        This result can be added to a vector to ensure it has no zeros (it will ... have
        ... SMALLESTFLOAT instead) and those places where there are no zeros ...
        ... remain the same.
        """
        A = self.A
        B = self.B
        N = self.N
        T = len(observations)
        O = self._get_observation_indices(observations)
        logPi = log( self.pi + equal( self.pi, 0.0 ) * SMALLESTFLOAT )
        logA = log( A + (equal( A, 0.0 ) * SMALLESTFLOAT) )
        logB = zeros( (self.M, N), float)
        for i in xrange(self.M):
            t = B[i, :]
            logB[i] = log( t + equal( t, 0.0 ) * SMALLESTFLOAT )

        # -- initialisation
        delta = add( logB[O[0]], logPi) 
        delta_t = zeros( N, float )
        psi = zeros( (T, N), int )

        # -- recursion
        tmp = zeros( N, float )
        for t in xrange( 1, T ):
            for j in xrange(N):
                tmp = delta + logA[:, j]
                psi[t, j] = tmp.argmax()
                delta_t[j] = tmp[psi[t, j]] + logB[O[t], j]
            delta, delta_t = delta_t, delta

        # -- termination
        index_star = [delta.argmax()]
        for psi_t in psi[-1:0:-1]:
            index_star.append( psi_t[index_star[-1]] )
        index_star.reverse()

        sequence = [self.STATES[i] for i in index_star]
        return sequence

    def observation_likelihood_simple(self, observations, Q):
        """likelihood of a sequence of observations being generated by a 
        given sequence of hidden states Q.
        
        *** This procedure doesn't take into consideration the
        transition probabilities, because said transitions are either
        known or considered known. """
        O = self._get_observation_indices(observations)
        states = [self.state_index[s] for s in Q]

        accu = 1
        for o, s in zip(O, states): accu *= self.B[o, s]
        return accu

    def observation_likelihood_log(self, observations, Q, asProbability = False):
        """log(likelihood) of a sequence of observations being generated by a 
        given sequence of hidden states Q.
        
        Just as observation_likelihood_simple but using logarithms."""
        O = self._get_observation_indices(observations)
        states = [self.state_index[s] for s in Q]

        N = self.N
        M = self.M
        logB = zeros( (M, N), float)
        for i in xrange(M):
            t = self.B[i, :]
            logB[i] = log(t + (equal(t, 0.0) * SMALLESTFLOAT)) 
                  #  (equal(t, 0.0) * SMALLESTFLOAT) => ...
                  # ... this trick is explained in viterbi_log
        accu = 0
        for o, s in zip(O, states): accu += logB[o, s]

        if asProbability: return e ** accu
        return accu

    def transition_likelihood_simple(self, Q, Ob):
        """likelihood of a sequence of hidden states Q and observations Ob
        (forward algorithm) """
        O = self._get_observation_indices(Ob)
        states = [self.state_index[s] for s in Q]

        accu = self.pi[states[0]] * self.B[O[0], states[0]]
        for i in xrange(1,len(Q)):
            accu *= self.A[states[i-1], states[i]] * self.B[O[i], states[i]]
            # Rabiner (17)
        return accu

    def transition_likelihood_log(self, Q, Ob, asProbability = False): # it could be useful to have logarithmic versions of the matrices as class members to speed things up.
        """log(likelihood) of a sequence of hidden states Q and observations Ob
        (forward algorithm) """
        O = self._get_observation_indices(Ob)
        states = [self.state_index[s] for s in Q]

        logPi = log( self.pi + equal( self.pi, 0.0 ) * SMALLESTFLOAT )
        logA = log( self.A + (equal( self.A, 0.0 ) * SMALLESTFLOAT) )
            #  (equal(A, 0.0) * SMALLESTFLOAT) => ...
            # ... this trick is explained in viterbi_log
        logB = zeros( (self.M, self.N), float)
        for i in xrange(self.M):
            t = self.B[i, :]
            logB[i] = log( t + equal( t, 0.0 ) * SMALLESTFLOAT )

        accu = logPi[states[0]] + logB[O[0], states[0]]
        for i in xrange(1,len(Q)):
            accu += logA[states[i-1], states[i]] + logB[O[i], states[i]]
        # as in Rabiner (17) but using log to avoid underflow

        if asProbability: return e ** accu
        return accu

    def _cap(self, norm = False, verbose = False, check = False):
        """ cap A,B,pi elements at 1 (if norm == False)
        capping is quicker but normalise will fix any rounding problems that may
        compromise the model's integrity in some situations."""
        if norm:
            normalise(self.A  , dimNorm=1)
            normalise(self.B  , dimNorm=0)
            normalise(self.pi , dimNorm=0)
        else:
            mask = self.A > 1
            self.A[mask] = 1.0
            mask = self.B > 1
            self.B[mask] = 1.0
            mask = self.pi > 1
            self.pi[mask] = 1.0
        if verbose: print "CAP"
        if check: self.integrity_check()

    def baum_welch_o(self, observations, maxiter = BAUMWELCH_MAXITER, verbose=True):
        """helper to call Baum-Welch translating observations into their indices"""
        O = self._get_observation_indices(observations)
        return self.baum_welch_in(O, maxiter, verbose)

    def baum_welch_in(self, obsIndices, maxiter=BAUMWELCH_MAXITER, verbose=True):
        """Uses Baum-Welch using scaling to avoid underflow (as described in Rabiner)
        Each iteration prints a dot on stderr, or a star if scaling was
        applied"""
        if DISPLAY_INTERVAL == 0: verbose = False
        Bo = take(self.B, obsIndices, 0) # makes a potentially massive matrix with a lot of redundancy
        # this is simple and, as far as I've tested, quite fast thanks to numpy. But I believe it would be
        # much better just passing the sequence of observation indices as a parameter. ***TODO***
        for iter in xrange(maxiter):
            alpha, scaling_factors = self.alpha_scaled(self.A, Bo, self.pi)
            beta = self.beta_scaled(self.A, Bo, scaling_factors)
            gamma = self._bw_gamma(alpha, beta, scaling_factors)
            A_bar, B_bar, pi_bar = self._bw_model_recalculation(gamma, self.ksi( self.A, Bo, alpha, beta ), obsIndices)
            if verbose and ((iter % DISPLAY_INTERVAL) == 0):
                loglikelihood = self._likelihood(scaling_factors)
                likelihood = e ** loglikelihood
                print "Iterations so far: ", (iter+1), " log likelihood=", self._likelihood(scaling_factors), " likelihood=",likelihood
            if self._bw_converged(A_bar, B_bar, pi_bar):
                iter +=1
                if verbose:
                    print 'Done in %d iterations' % iter
                    loglikelihood = self._likelihood(scaling_factors)
                    likelihood = e ** loglikelihood
                    print " log likelihood=", self._likelihood(scaling_factors), " likelihood=",likelihood
                return iter
            else:
                normalise(A_bar,  dimNorm = 1)
                normalise(B_bar,  dimNorm = 0)
                normalise(pi_bar, dimNorm = 0)
                self.A  = A_bar
                self.B  = B_bar
                self.pi = pi_bar
        if verbose: print "Baum-Welch did not converge in %d iterations" % maxiter # note that Baum-Welch was run anyway and the matrices changed
        return maxiter

    def _likelihood( self, scaling_factors ):
        """Logarithmic likelihood of the training set using the precomputed
        alpha probabilities (sum(k=0..N,alpha(T,k)).
        As long as there is convergence it will increase every iteration in Baum-Welch."""
        t = where( scaling_factors==0.0, SMALLESTFLOAT, scaling_factors )
        return -add.reduce( log(t) )

    @staticmethod
    def alpha_scaled(A, Bo, pi):
        """alpha(forward) from the forward-algorithm, using rescaling
        (as detailed in Rabiner).
    
        Bo is the "slice" of the observation probability matrix corresponding
        to the observations (ie Bo=take(B,observation_indices)).
        For each t, c(t)=1./sum(alpha(t,i)), and C(t)=product(k=0..t,c(t))
        and alpha_scaled(t,i)=alpha(t,i)*C(t)
        The function returns: (alpha_scaled,C(t))
        """
        T = Bo.shape[0]
        N = A.shape[0]
        alpha = Bo[0] * pi # alpha initialisation (formula 19 in Rabiner) 
        scaling_factors = zeros(T, float)
        scaling_factors[0] = 1./add.reduce(alpha)     # scaling factors at t=0  
        alpha_scaled = zeros((T, N), float)
        alpha_scaled[0] = alpha * scaling_factors[0]
        for t in xrange(1, T):
            alpha = dot(alpha_scaled[t-1], A) * Bo[t] # matrix multiplication (induction step) (formula 92a in Rabiner)      
            scaling_t = 1./add.reduce(alpha)
            scaling_factors[t] = scaling_t            # scaling factors are stored for every t
            alpha_scaled[t] = alpha * scaling_t       # scaled coefficients  (formula 92b in Rabiner) 
        return alpha_scaled, scaling_factors

    @staticmethod
    def beta_scaled( A, Bo, scaling_factors ):
        """beta(backward) from the backward-algorithm, using rescaling
        (as detailed in Rabiner).
    
        beta(t,i)=P(O(t+1),...,O(T),Q(t)=Si|model)
        Or if scaling_factors is not None:
        beta_scaled(t,i)=beta(t,i)*C(t) (From the result of _alpha_scaled)
        Bo is the same as in function _alpha
        """
        T, N = Bo.shape
        assert N == A.shape[0]
        scaling_factors = scaling_factors
        beta = zeros((T, N), float)
        tmp = zeros(N, float)
        beta[-1] = ones(N, float) * scaling_factors[-1] #  (formula 24 in Rabiner) 
        for t  in xrange(T-2, -1, -1):
            multiply(scaling_factors[t], Bo[t+1], tmp)
            multiply(tmp, beta[t+1], tmp)
            beta[t] = dot(A, tmp )    #  (formula 25 in Rabiner) 
        return beta

    @staticmethod
    def ksi( A, Bo, alpha, beta ):
        """ (Rabiner 36) ===> ksi(t,i,j)=P(q_t=Si,q_(t+1)=Sj|lambda) """
        N = A.shape[0]
        T = len(Bo)
        ksi_ = zeros( (T-1, N, N), float ) # ksi_ because ksi is already defined in this context (method name)
        tmp = Bo * beta
        for t in range(T-1):  # transpose[alpha] . (B[obs]*beta[t+1])  # (. => matrix product)
            ksit = ksi_[t, :, :]
            multiply( A, tmp[t+1], ksit ) # remember this does modify ksi_ as np works by reference when assigning (and most of the time for that matter)
            multiply( ksit, alpha[t, :, newaxis], ksit )
            ksi_sum = add.reduce( ksit.flat )
            ksit /= ksi_sum
        return ksi_


    def baum_welch_multiple(self, m_observations,
                       maxiter = BAUMWELCH_MAXITER, verbose=True ):
        """Uses Baum-Welch algorithm to learn the probabilities on multiple
        observations sequences
        """
        # remove empty lists
        m_observations = filter( lambda x: x, m_observations )
        setO =  set()   # set of obsevations        
        K = len( m_observations )
        sigma_gamma_A = zeros((self.N, ), float)
        sigma_gamma_B = zeros((self.N, ), float)
        A_bar  = zeros((self.N, self.N), float)
        B_bar  = zeros((self.M, self.N), float)
        pi_bar = zeros(self.N, float)
        if DISPLAY_INTERVAL == 0: #DISPLAY_INTERVAL
            dispiter = maxiter
        else:
            dispiter = DISPLAY_INTERVAL
        obs_list = []
        for k in range(K):
            observations = m_observations[k]
            obsIndices = self._get_observation_indices(observations)
            obs_list.append( obsIndices )
            setO = setO | set(observations)  # add new elements observed
        for iter in xrange( 1, maxiter + 1 ):
            total_likelihood = 0
            for k in range(K):
                obsIndices = obs_list[k]
                Bo = take(self.B, obsIndices, 0)
                alpha, scaling_factors = self.alpha_scaled( self.A, Bo, self.pi )
                beta  = self.beta_scaled( self.A, Bo, scaling_factors )
                ksy   = self.ksi( self.A, Bo, alpha, beta )
                gamma = self._bw_gamma( alpha, beta, scaling_factors )
                pi_bar += gamma[0]
                sigma_gamma_kA = add.reduce(gamma[:-1])
                sigma_gamma_A += sigma_gamma_kA       # (109) (110) denominator
                sigma_gamma_B += sigma_gamma_kA + gamma[-1]
                A_bar_k = add.reduce( ksy )
                add( A_bar, A_bar_k, A_bar )           # (109) numerator
                for j in xrange(len(obsIndices)):     # (numerator in Rabiner 110)
                    B_bar[obsIndices[j]] += gamma[j]

                total_likelihood += self._likelihood( scaling_factors )
                
            #end for k in range(K)

            # sigma_gamma(i)=0 implies A(i,:)=0 and B(i,:)=0
            sigma_gamma_A = 1. / where( sigma_gamma_A, sigma_gamma_A, 1 )
            A_bar *= sigma_gamma_A[:, newaxis]    # (Rabiner 109)

            sigma_gamma_B = 1./where( sigma_gamma_B, sigma_gamma_B, 1) # inverted avoiding division by 0
            B_bar *= sigma_gamma_B    # (see Rabiner 110)
            pi_bar /= K
            _bwm_correct_matrix(A_bar, 1, 1. / self.N, False)
            _bwm_correct_matrix(B_bar, 0, 1. / self.M, False)
            if verbose and ((iter % dispiter) == 0): print "Iterations so far: ", iter, " log total likelihood :", total_likelihood
            if self._bw_converged(A_bar, B_bar, pi_bar):
                if verbose: print 'Done in %d iterations' % iter
                break
            self.A, A_bar   = A_bar, self.A
            self.B, B_bar   = B_bar, self.B
            self.pi, pi_bar = pi_bar, self.pi
            A_bar.fill(0)
            B_bar.fill(0)
            pi_bar.fill(0)
            sigma_gamma_A.fill(0)
            sigma_gamma_B.fill(0)
        else:
            if verbose:
                print "The Baum-Welch algorithm did not converge in",
                print " %d iterations" % maxiter
        self._cap()
        # Correct B in case 0 probabilities slipped in
        setO = set(self.OBSERVATIONS) - setO
        while setO != set():
            e = setO.pop()
            e = self._get_observation_indices([e])
            self.B[e[0]] = 0
        return iter

    def _bw_model_recalculation( self, gamma, ksi, obsIndices ): # move out?
        """model recalculation. Baum-Welch last step in every iteration"""
        sigma_gamma_A = add.reduce(gamma[:-1])
        sigma_gamma_B = add.reduce(gamma)
        for i in range(len(sigma_gamma_B)):
            if sigma_gamma_B[i] < EPSILON:  sigma_gamma_B[i] = 1
        for i in range(len(sigma_gamma_A)):
            if sigma_gamma_A[i] < EPSILON:  sigma_gamma_A[i] = 1
        pi_bar = gamma[0]  # new PI (Rabiner 40a)
        A_bar  = add.reduce(ksi)
        A_bar /= sigma_gamma_A[:, newaxis] # new A (Rabiner 40b)       
        B_bar = zeros( (self.M, self.N), float )
        for i in xrange( len(obsIndices) ):
            B_bar[obsIndices[i]] += gamma[i] 
        B_bar /= sigma_gamma_B # new B (Rabiner 40c)
        return A_bar, B_bar, pi_bar

    def _bw_converged( self, A, B, pi ): # move out?
        """if the difference between the estimated model
        and the current model is small enough, Baum-Welch stops"""
        return (allclose( self.A, A, alpha_RTOL, alpha_ATOL) and 
               allclose( self.B, B, beta_RTOL, beta_ATOL) and 
               allclose( self.pi, pi, pi_RTOL, pi_ATOL))
    
    def _bw_gamma(self, alpha, beta, scaling_factors ):  # move out?
        """ (Rabiner 26, 27) ==>  gamma_t(i)=P(q_t=Si|lambda)"""
        g = alpha * beta / scaling_factors[:, newaxis]
        return g


    def _weighting_factor_Pall(self, setObs):
        """compute Wk = P(setObservations | lambda_k) """
        P = 1
        for obs in setObs:
            Tk = len(obs)
            obsIndices = self._get_observation_indices(obs)
            Bo = take(self.B, obsIndices, 0)
            null = 0
            for i in range(Tk):
                null = null or (allclose(Bo[i], zeros([self.N])))
            if null:
                P = 0
            else:
                alpha_s, scalingFactor = self.alpha_scaled(self.A, Bo, self.pi)
                alpha = alpha_s[Tk-1] / product(scalingFactor, 0) 
                P *= add.reduce(alpha)
        return P

    def _weighting_factor_Pk(self, observation):
        """compute Wk = P(Observation_k | lambda_k) """
        Tk = len(observation)
        obsIndices = self._get_observation_indices(observation)
        Bo = take(self.B, obsIndices, 0)
        alpha_s, scalingFactor = self.alpha_scaled(self.A, Bo, self.pi)
        alpha = alpha_s[Tk-1] / product(scalingFactor, 0)
        return add.reduce(alpha)

    def ensemble_averaging(self, setObservations, weighting_factor="unit", 
                            maxiter=ENSEMBLE_MAXITER, verbose=True):
        """Uses ensemble averaging method to learn the probabilities on multiple
        observation sequences"""
        N = self.N
        W = 0
        self._cap()
        #hmmk = HMM(self.STATES, self.OBSERVATIONS, self.A, self.B, self.pi)
        hmmk = deepcopy(self)
        A_bar = zeros((N, N))
        B_bar = zeros((self.M, N))
        pi_bar = zeros(N)
        for obs in setObservations:
            hmmk.A = self.A
            hmmk.B = self.B
            hmmk.pi = self.pi
            hmmk.baum_welch_o(obs, maxiter, verbose)
            if weighting_factor == "Pall":
                Wk = hmmk._weighting_factor_Pall(setObservations)
            elif weighting_factor == "Pk":
                Wk = hmmk._weighting_factor_Pk(obs)
            else:
                Wk = 1
            A_bar = A_bar + Wk * hmmk.A
            B_bar = B_bar + Wk * hmmk.B
            pi_bar = pi_bar + Wk * hmmk.pi
            W = W + Wk
        if W == 0:
            W = 1
            print "The ensemble averaging method did not converge"
        else:
            if verbose:
                print "W:",W
                print A_bar
                print B_bar
                print pi_bar
            self.A = A_bar / W
            self.B = B_bar / W
            self.pi = pi_bar / W
            self._cap()

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Mixin class for dealing with HMM when we can train with knowledge of hidden
# states.
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


class HMMS_mixin: 
    def _baum_welch_multiple_learn_PI(self, setStates):
        """learn PI according to given known samples of sequences of hidden states"""
        len_s = len(setStates)
        counters = {}
        for i in setStates: counters[i[0]] = counters.get(i[0], 0) + 1
        for i in xrange(self.N): self.pi[i] = counters.get(self.STATES[i], 0) / float(len_s)
        self.integrity_check()

    def _learn_A(self, states):
        """learn A according to a given known sequence of hidden states"""
        T = len(states)
        N = self.N
        self.A = zeros((N, N))
        for k in range(T-1):
            Si = states[k]
            Sj = states[k+1]
            i = self.state_index[Si]
            j = self.state_index[Sj]
            self.A[i, j] += 1
        for i in range(N):
            if add.reduce(self.A, 1)[i]==0:
                self.A[i, i] = 1
        self.A *= 1./add.reduce(self.A, 1)[:, newaxis]

    def _baum_welch_multiple_learn_A(self, setStates):
        """learn A according to given known samples of sequences of hidden states"""
        h = HMMS(self.STATES, [1])
        A_bar = zeros((self.N, self.N), float)
        h.A = self.A
        for seq in setStates:
            h._learn_A(seq)
            A_bar += h.A
        self.A = A_bar / len(setStates)

    def baum_welch_o(self, observations, states, maxiter = BAUMWELCH_MAXITER, verbose=True):
        """helper to call Baum-Welch translating observations into their indices"""
        O = self._get_observation_indices(observations)
        return self.baum_welch_in(O, states, maxiter, verbose)

    def baum_welch_in( self, obsIndices, states, maxiter=BAUMWELCH_MAXITER, verbose=True ):
        """Uses Baum-Welch using scaling to avoid underflow (as described in Rabiner)
        Each iteration prints a dot on stderr, or a star if scaling was
        applied"""
        if verbose:
            print "called baum_welch_in:", obsIndices, states
        self._learn_A(states)
        if DISPLAY_INTERVAL == 0: verbose = False
        Bo = take(self.B, obsIndices, 0)
        for iter in xrange(maxiter):
            alpha, scaling_factors = self.alpha_scaled(self.A, Bo, self.pi)
            beta = self.beta_scaled(self.A, Bo, scaling_factors)
            gamma = self._bw_gamma(alpha, beta, scaling_factors)
            B_bar, pi_bar = self._bw_model_recalculation(gamma, obsIndices)
            if verbose and ((iter % DISPLAY_INTERVAL) == 0):
                loglikelihood = self._likelihood(scaling_factors)
                likelihood = e ** loglikelihood
                print "Iterations so far: ", (iter+1), " log likelihood=", self._likelihood(scaling_factors), " likelihood=",likelihood
            if self._bw_converged(B_bar, pi_bar):
                iter += 1
                #print "#123", self.B
                _bwm_correct_matrix(self.B, 0, 1. / self.M, False)
                normalise(self.B,  dimNorm = 0)
                #print "#124", self.B
                if verbose:
                    print 'Done in %d iterations' % iter
                    loglikelihood = self._likelihood(scaling_factors)
                    likelihood = e ** loglikelihood
                    print " log likelihood=", self._likelihood(scaling_factors), " likelihood=",likelihood
                return iter
            else:
                #normalise(B_bar,  dimNorm = 0)
                #normalise(pi_bar, dimNorm = 0)
                self.B  = B_bar
                self.pi = pi_bar
        if verbose:
            print "The Baum-Welch algorithm did not converge"
            print " in %d iterations" % maxiter
            _bwm_correct_matrix(self.B, 0, 1. / self.M, False)        
            normalise(self.B,  dimNorm = 0)
        return maxiter


    def baum_welch_multiple(self, m_observations, setStates,
                       maxiter = BAUMWELCH_MAXITER, verbose = True ):
        """Uses Baum-Welch algorithm to learn the probabilities on multiple
        observations sequences and states sequences
        """
        # remove empty lists
        m_observations = filter( lambda x: x, m_observations )
        setO =  set()   # set of obsevations        
        K = len( m_observations )
        sigma_gamma_B = zeros((self.N, ), float)
        ##A_bar  = zeros((self.N, self.N), float)
        B_bar  = zeros((self.M, self.N), float)
        ##pi_bar = zeros(self.N, float)
        self._baum_welch_multiple_learn_A(setStates)
        self._baum_welch_multiple_learn_PI(setStates)
        if DISPLAY_INTERVAL == 0:
            dispiter = maxiter
        else:
            dispiter = DISPLAY_INTERVAL
        obs_list = []
        for k in range(K):
            observations = m_observations[k]
            obsIndices = self._get_observation_indices(observations)
            obs_list.append( obsIndices )
            setO = setO | set(observations)  # add new elements observed
        for iter in xrange( 1, maxiter + 1 ):
            total_likelihood = 0
            for k in range(K):
                obsIndices = obs_list[k]
                Bo = take(self.B, obsIndices, 0)
                alpha, scaling_factors = self.alpha_scaled( self.A, Bo, self.pi )
                beta  = self.beta_scaled( self.A, Bo, scaling_factors )
                gamma = self._bw_gamma( alpha, beta, scaling_factors )
                ##pi_bar += gamma[0]
                sigma_gamma_k = add.reduce(gamma[:-1])
                sigma_gamma_B += sigma_gamma_k + gamma[-1]

                for j in xrange(len(obsIndices)):     # (numerator in Rabiner 110)
                    B_bar[obsIndices[j]] += gamma[j]
                total_likelihood += self._likelihood( scaling_factors )
                
            sigma_gamma_B = 1./where(sigma_gamma_B, sigma_gamma_B, 1)
            B_bar *= sigma_gamma_B    # (110)
            ##pi_bar /= K
            _bwm_correct_matrix(B_bar, 0, 1. / self.M, False)
            if verbose and ((iter % dispiter) == 0): print "Iterations so far: ", iter, " log total likelihood :", total_likelihood
            ##if self._bw_converged(B_bar, pi_bar):
            if self._bw_converged_b(B_bar):
                if verbose: print 'Done in %d iterations' % iter
                break
            
            self.B, B_bar   = B_bar, self.B
            ##self.pi, pi_bar = pi_bar, self.pi
            ##A_bar.fill(0)
            B_bar.fill(0)
            ##pi_bar.fill(0)
            sigma_gamma_B.fill(0)
        else:
            if verbose:
                print "The Baum-Welch algorithm did not converge",
                print " in %d iterations" % maxiter
        self._cap()
        # Correct B in case 0 probabilities slipped in
        setO = set(self.OBSERVATIONS) - setO
        while setO != set():
            e = setO.pop()
            e = self._get_observation_indices([e])
            self.B[e[0]] = 0
        return iter

    def _bw_converged(self, B_bar, pi_bar):
        """Returns true if the difference between the estimated model
        and the current model is small enough that we can stop the
        learning process"""
        return (allclose( self.pi, pi_bar, pi_RTOL, pi_ATOL) and 
               allclose( self.B, B_bar, beta_RTOL, beta_ATOL))

    def _bw_converged_b(self, B_bar):
        """Returns true if the difference between the estimated model
        and the current model is small enough that we can stop the
        learning process"""
        return (allclose( self.B, B_bar, beta_RTOL, beta_ATOL))

    def _bw_model_recalculation( self, gamma, obsIndices ):
        """Compute the new model, using gamma"""
        sigma_gamma_B = add.reduce(gamma)
        for i in range(len(sigma_gamma_B)):
            if sigma_gamma_B[i] < EPSILON:
                sigma_gamma_B[i] = 1
        ## Compute new PI
        pi_bar = gamma[0]                       # (40a)
        
        ## Compute new B
        B_bar = zeros( (self.M, self.N), float )
        for i in xrange( len(obsIndices) ):
            B_bar[obsIndices[i]] += gamma[i]
        B_bar /= sigma_gamma_B

#        ## >>> DEBUG
#        print sigma_gamma_B.shape, sigma_gamma_B
#        for i in sigma_gamma_B:
#            if i == 0:
#                print "ERROR *****", sigma_gamma_B
#                print gamma
#                print obsIndices
#                print "***************************"
#                break
#        for i in B_bar:
#            for j in i:
#                if not(j >= 0 and j<=1):
#                    print "ERROR *****", sigma_gamma_B
#                    print gamma
#                    print obsIndices
#                    print "***************************"
#                    break
#        ## <<< DEBUG

        return B_bar, pi_bar
    
    def ensemble_averaging(self, setObservations, setStates, 
                        weighting_factor="unit", maxiter=ENSEMBLE_MAXITER, verbose=True):
        """Uses ensemble averaging method to learn the probabilities on 
        multiple observations sequences and states sequences"""
        N = self.N
        W = 0
        #hmmk = self.__class__(self.STATES, self.OBSERVATIONS)
        hmmk = deepcopy(self)        
        A_bar = zeros((N, N))
        B_bar = zeros((self.M, N))
        pi_bar = zeros(N)
        for k, obs in enumerate(setObservations):
            hmmk.A = self.A
            hmmk.B = self.B
            hmmk.pi = self.pi
            state = setStates[k]
            if verbose: print "pre hmmk", hmmk
            hmmk.baum_welch_o(obs, state, maxiter, verbose)
            #print "postttttt"
            #_bwm_correct_matrix(hmmk.B, 0, 1. / hmmk.M, False)
            #print "postttttt*2"
            if weighting_factor == "Pall":
                Wk = hmmk._weighting_factor_Pall(setObservations)
            elif weighting_factor == "Pk":
                Wk = hmmk._weighting_factor_Pk(obs)
            else:
                Wk = 1
            A_bar = A_bar + Wk * hmmk.A
            B_bar = B_bar + Wk * hmmk.B
            pi_bar = pi_bar + Wk * hmmk.pi
            W = W + Wk
            #print "#ensemble",hmmk
            #print "#223", hmmk.B
            _bwm_correct_matrix(hmmk.B, 0, 1. / hmmk.M, False)   
            #print "#224", hmmk.B
            normalise(hmmk.B,  dimNorm = 0)
            ##normalise(pi_bar, dimNorm = 0)
            #print "#225", hmmk.B
            if verbose: print "#ensemble",hmmk
        if W == 0:
            W = 1
            print "The ensemble averaging method did not converge" 
        else:
            if verbose:
                print "W:",W
                print A_bar
                print B_bar
                print pi_bar
            self.A = A_bar / W
            self.B = B_bar / W
            self.pi = pi_bar / W
            self._cap()

class HMMS(HMMS_mixin, HMM):
    pass


