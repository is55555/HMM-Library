import time
from numpy.linalg import norm
from copy import deepcopy

#def geometic_mean(numbers):
#    return (reduce(lambda i, j: i*j, numbers))**(1.0/len(numbers))
def geometric_mean(numbers):
    product = 1.
    for n in numbers: product *= n
    return product ** (1.0/len(numbers))

def arithmetic_mean(numbers):
    sum = 0.
    for n in numbers: sum += n
    return float(sum) / len(numbers)

def frobenius_error(_reference, _test, normalise_models = True):
    if normalise_models:
        reference = deepcopy(_reference) # so as not to modify the reference model
        test = deepcopy(_test)           # if we wanted to modify them, ...
                # ... we'd call normalise_hmm_by_reachability on them ...
                # ... before calling this function, and then we'd call ...
                # ... this function with normalise_models set to False.
        reference.normalise_hmm_by_reachability()
        test.reorder_to_match(reference.STATES)
    else:
        reference = _reference
        test = deepcopy(_test)
        test.reorder_to_match(reference.STATES)

    error1 = norm(reference.A  - test.A)
    error2 = norm(reference.B  - test.B)
    error3 = norm(reference.pi - test.pi)
    return error1, error2, error3


verbose=1

def set_verbose( v ):
    global verbose
    verbose = v

def show_analysis(hmm, chain):
    if verbose:
        print "Chain      : ", chain
        print "analyse    : ", hmm.viterbi(chain)
        print "analyse_log: ", hmm.viterbi_log(chain)

# try to run the call for at most 2sec.
AVG = 2.

def time_calls(func, args, warmUp = False, givenTime = 5, minCalls = 2, maxCalls = 10000):
    """ time_calls(func, args, warmUp = False, givenTime = 5, minCalls = 2, maxCalls = 10000)
    Note that results of the called function aren't kept. It would make sense to keep some or
    all results in a dictionary (for instance) but that would complicate the interface somewhat.
    Might add a tentative version for this. """
    if warmUp: func(*args)
    t0 = time.time()
    func(*args)
    t1 = time.time()
    if t1 == t0: #          print t0, t1, "zero time call?"
            return (0,0,1)
    elif maxCalls == 1:
            t = t1-t0
            return(t,t,1)

    ncalls = int(givenTime / (t1 - t0))
    if ncalls < minCalls: ncalls = minCalls
    if ncalls > maxCalls: ncalls = maxCalls # not in an elif, just in case maxCalls < minCalls. The
           # upper limit takes precedence
    accumulatedTime = 0.0
    min = t1 - t0
    for i in xrange(ncalls):
        t0 = time.time()
        func(*args)
        t1 = time.time()
        t = t1 - t0
        accumulatedTime += t
        if t < min: min = t
    avg = accumulatedTime / ncalls
    # print "%s: avg = %8.2f%s ; min = %8.2f%s ; runs = %d" % (desc, avg, un,  min, un, ncalls)
    return (avg, min, ncalls)

def timecall(description, func, args):
    print description,":", "average = %8.6f s, min = %8.6f s, trials = %d" % time_calls(func, args)

def timecall_one(description, func, args):
    print description,":", "average = %8.6f s, min = %8.6f s, trials = %d" % time_calls(func, args, maxCalls = 1)


def norm2(m):
    """Returns the norm2 of a matrix"""
    v = reshape(m, (product(m.shape)))
    return sqrt(dot(v, v)) / product(m.shape)

STATES = "abcdefghijklmnopqrstuvwxyz"
VALUES = [ "s%02d" % i for i in range(100) ]

def deterministic_hmm_gen( NSTATES = (2,4,10,15,20),
                           NVALUES = range(5,100,10)
                           ):
    """Generates 5-tuples descriptions of various
    state-deterministic HMM
    """
    for nstate in (2, 4, 10, 15, 20):
        for nobs in range(5, 100, 10):
            states = list(STATES[:nstate])
            values = list(VALUES[:nobs])
            ID = identity(nstate)
            A = concatenate((ID[1:, :], ID[0:1, :] ), 0)
            pi = zeros( (nstate), float )
            pi[0] = 1.
            B = zeros( (nobs, nstate), float )
            bi = zeros( (nobs), float )
            for k in range(nstate):
                bi[:] = .5 / (nobs - 1)
                bi[k] = .5
                B[:, k] = bi
            yield states, values, A, B, pi
