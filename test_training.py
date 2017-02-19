from copy import deepcopy
from time import time
from numpy import take

from support import arithmetic_mean, frobenius_error # moved away from geometric mean ...
   # ... because it could be down to 0 if some edge cases where we get a perfect match...
   # ... in one of the matrices (like PI for instance) - this can happen in small models ...
   # ... relatively easily.

from hmm import HMM, HMMS
from precisionParameters import ATOL, RTOL

from sample_hmm import deterministic_hmm

def check_viterbi_likelihoods(length, h = None, verbose = False):
  """ tests we obtain a more likely model of state transitions by using viterbi
  on a sequence of observations than the actual state transitions that created
  said observation sequence."""
  if h == None:
    h = HMM(['sunny','rainy','storm','cloudy','foggy'],['dry','damp','wet','soaked'])
    h.random_initialisation()
  if verbose:
    print h
    print "---------------"
  sim = h.simulate(length,True)
  Q,O = zip(*sim)
  _ls = h.transition_likelihood_simple(Q, O)
  _ll = h.transition_likelihood_log(Q, O, True)
  if verbose:
    print "actual path         :", Q
    print "actual observations :", O
    print "likelihood (simple) %.40f" % _ls, _ls
    print "likelihood (log   ) %.40f" % _ll, _ll
    print "---------------"
  vd = h.viterbi(O)
  vl = h.viterbi_log(O)
  assert len(O) == len(vd) == len(vl)
  
  _lvs = h.transition_likelihood_simple(vd, O)
  _lvl = h.transition_likelihood_log(vl, O, True)

  if verbose:
    print "viterbi (direct)", vd
    print "likelihood (simple) %.40f" % _lvs, _lvs
    print "viterbi (log   )", vl
    print "likelihood (log   ) %.40f" % _lvl, _lvl
    print "---------------"
    print "---------------"
    print "random walks (100)"  # only done in verbose mode
    ac = 0
    min = 1. ; max = 0.
    for i in xrange(100):
      rw = h.random_walk(length)
      #print "random walk", rw
      lh = h.transition_likelihood_simple(rw, O)
      if lh>max: max = lh
      if lh<min: min = lh
      ac+=lh
      #print lh , 
    print "\naverage", (ac / 100), "min:", min, "max:", max

  assert _lvl >= _ll, str(O) + str(vl) + str(_ll) +" <-ll lvl-> "+ str(_lvl) # the likelihood of the transitions for the... 
  assert _lvs >= _ls #...viterbi-obtained sequence must be >= the...
  # ... likelihood of the actual path from the simulation.


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

def test_baum_welch_multiple(number_obs, len_obs, initialisation = "random"):
    print "Baum-Welch from multiple observations."
    print number_obs,"observations of", len_obs,"length. Init mode:",initialisation
    h = HMM(['sunny','rainy','storm','cloudy','foggy'],['dry','damp','wet','soaked'])
    h.random_initialisation()
    chains = []
    for i in xrange(number_obs): chains.append(h.simulate(len_obs))
    h2 = deepcopy(h)
    
    if initialisation.lower() == "equiprobable": h.equiprobable_initialisation()
    elif initialisation.lower() in ["no", "none"]: pass
    else: h.random_initialisation()

    h.baum_welch_multiple(chains)
    print "---- generated ----"
    print h
    print "---- original  ----"
    print h2
    h.integrity_check()
    h2.integrity_check()

def test_baum_welch_multiple_s(number_obs, len_obs, initialisation = "random"):
    print "Baum-Welch from multiple observations with hidden state training."
    print number_obs,"observations of", len_obs,"length. Init mode:",initialisation
    h = HMMS(['sunny','rainy','storm','cloudy','foggy'],['dry','damp','wet','soaked'])
    h.random_initialisation()
    chains_O = []
    chains_Q = []
    for i in xrange(number_obs):
        sim = h.simulate(len_obs, True)
        Q, O = zip(*sim)
        chains_O.append(O)
        chains_Q.append(Q)
    h2 = deepcopy(h)
    
    if initialisation.lower() == "equiprobable": h.equiprobable_initialisation()
    elif initialisation.lower() in ["no", "none"]: pass
    else: h.random_initialisation()

    h.baum_welch_multiple(chains_O, chains_Q)
    print "---- generated ----"
    print h
    print "---- original  ----"
    print h2
    h.integrity_check()
    h2.integrity_check()

def viterbi_likelihoods_suite():
  check_viterbi_likelihoods(25, verbose = True)
  print "******* testing for the odd chance of viterbi losing to the observation"
  print "(this seldom happens as I type this because of a sneaky bug: in the "
  print "likelihood function I'm only considering A and B but not PI, so it could "
  print "happen that the viterbi-calculated sequence is not as good as the actual "
  print "sequence according to this buggy likelihood function, however it doesn't "
  print "happen often. In 1000 trials and with a short-ish sequence and a random "
  print "HMM, it should arise almost every time.)"
  print "**********************************************************************"
  for i in xrange(10):
    print i,
    check_viterbi_likelihoods(25)
  print "**********************************************************************"
  print "**********************************************************************"
  #test_baum_welch_multiple  ( 500,  6,  initialisation = "none") #  
  #test_baum_welch_multiple  ( 500, 20,  initialisation = "equiprobable") #  equiprobable always converge extremely quickly (and A doesn't change at all)
  #test_baum_welch_multiple  ( 500, 20,  initialisation = "random") #
  test_baum_welch_multiple  ( 250,  5,  initialisation = "random")
  test_baum_welch_multiple_s( 250,  5,  initialisation = "random")


# *****************************************************************************
# *****************************************************************************




#test = HMM(['s' + str(i) for i in range(1, nStates+1)],
#           ['o' + str(i) for i in range(1, nObs+1)])


def simple_baum_welch_test(nStates, nObs, simLength):
    print "\n\n\n\n\n##### simple_baum_welch_test",
    print "(nStates = %d, nObs = %d, simLength = %d): " % \
           (nStates,      nObs,      simLength)
    test = HMM(['s' + str(i) for i in range(1, nStates+1)],
               ['o' + str(i) for i in range(1, nObs+1)])
    test.random_initialisation()
    print "simple baum-welch.", nStates, "states", nObs, "symbols"
    print "Original"
    print test
    print
    print "sample simulation. Length:", simLength
    sample = test.simulate(simLength)
    print "randomising model..."
    test.random_initialisation()
    print "baum-welch on the single simulation..."
    test.baum_welch_o(sample)
    print "result:"
    print test


def multiple_baum_welch_test(nStates, nObs, nSims, simLength):
    print "\n\n\n\n\n##### multiple_baum_welch_test",
    print "(nStates = %d, nObs = %d, nSims = %d, simLength = %d): " % \
           (nStates,      nObs,      nSims,      simLength)
    test = HMM(['s' + str(i) for i in range(1, nStates+1)],
               ['o' + str(i) for i in range(1, nObs+1)])
    test.random_initialisation()
    print "simple baum-welch.", nStates, "states", nObs, "symbols"
    print "Original"
    print test
    print
    print "sample simulations.", nSims, "simulations *", simLength,"length"
    sample = test.simulate(simLength)
    l = []
    for i in range(nSims):
        obs = test.simulate(simLength)
        l.append(obs)
    print "randomising model..."
    test.random_initialisation()
    print "baum-welch on the generated simulations..."
    test.baum_welch_multiple(l)
    print "result:"
    print test

def deterministic_test_bw_simple(tries = 10):
    print "\n\n\n\n\n##### deterministic_test_bw_simple (tries = %d): " % tries
    reference = deterministic_hmm
    test = deepcopy(reference)
    test.equiprobable_initialisation() # first test will run on this setup 
                                    # (guaranteed to be bad, as A won't converge)
    data = reference.simulate(500)
    # note that all tries are performed on the same data    
    print "reference", reference
    errors_mean = []
    error_min = arithmetic_mean(frobenius_error(reference, test))
    best = deepcopy(test)
    for i in xrange(tries):
        print "\n\n\n\n\n***** loop",i
        t1 = time()
        iterations = test.baum_welch_o(data)
        t2 = time()
        error = frobenius_error(reference, test)
        error_mean = arithmetic_mean(error)
        print test
        print "try #", i, "BW iterations:", iterations
        print "error measure          :", error
        print "error (mean) :", error_mean
        print "time:", (t2 - t1)
        print "time per iteration:", (t2 - t1) / iterations
        errors_mean.append(error_mean)
        if error_mean < error_min:
            error_min = error_mean
            best = deepcopy(test)
        test.random_initialisation() # first iteration is done on the equiprobable
    print "\n\n\n\nBEST:"
    print best
    print "error_mean:",error_min
    print "error_measure:", frobenius_error(reference, best)

    print "## errors across all iterations (list of gm of frobenius)",
    print errors_mean
    print "Mean:", arithmetic_mean(errors_mean)
    print "sorted list of gm of frobenius", sorted(errors_mean)

def deterministic_test_bw_multiple(tries = 10): 
    print "\n\n\n\n\n##### deterministic_test_bw_multiple (tries = %d): " % tries
    reference = deterministic_hmm
    data = [reference.simulate(20) for i in xrange(20)] # 20 simulations 20-long each
    # note that all tries are performed on the same data
    test = deepcopy(reference)
    test.equiprobable_initialisation()
    errors_mean = []
    error_min = arithmetic_mean(frobenius_error(reference, test))
    best = deepcopy(test)
    for i in xrange(tries):
        print "\n\n\n\n\n***** loop", i
        t1 = time()
        iterations = test.baum_welch_multiple(data)        
        t2 = time()
        error = frobenius_error(reference, test)
        error_mean = arithmetic_mean(error)
        print test
        print "try #", i, "BWM iterations:", iterations
        print "error measure          :", error
        print "error (mean) :", error_mean
        print "time:", (t2 - t1)        
#        _A, _B, _pi = test.normalise_hmm()
        if error_mean < error_min:
            error_min = error_mean
            best = deepcopy(test)
        errors_mean.append(error_mean)
        test.random_initialisation() # first iteration is done on the equiprobable
    print "\n\n\n\nBEST:"
    print best
    print "error_mean:",error_min
    print "error_measure:", frobenius_error(reference, best)

    print "## errors across all iterations (list of gm of frobenius)",
    print errors_mean
    print "Mean:", arithmetic_mean(errors_mean)
    print "sorted list of gm of frobenius", sorted(errors_mean)


def _model_likelihood(model, data):
    likelihood = []
    for obs in data:
        obsIndices = model._get_observation_indices(obs)
        Bo = take(model.B, obsIndices, 0)
        alpha, scaling_factors = model.alpha_scaled(model.A, Bo, model.pi)
        likelihood.append(model._likelihood(scaling_factors))
    return arithmetic_mean(likelihood)
    
def deterministic_test_ensemble(tries=10, max_iter = 2000):
    print "\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
    print   "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
    print "##### determistic_test_ensemble",
    print "(tries = %d, max_iter = %d): " % (tries, max_iter)
    reference = deterministic_hmm
    reference.normalise_hmm_by_reachability() # not necessary in this case
    data = [reference.simulate(20) for i in xrange(20)] # 20 simulations 20-long each
    test = deepcopy(reference) # should make an "empty_like" method for cases like this
    test.equiprobable_initialisation()
    test._cap(norm=True) # ?  missed an integrity check in the deepcopy just below (rare?)
    error_vec = error_vec_min = frobenius_error(reference, test)
    error_sca_min = arithmetic_mean(error_vec_min)
    best = deepcopy(test)
    most_likely_measure = _model_likelihood(best, data)
    most_likely_model   = deepcopy(best)
    likelihood_list = []
    errors_mean = { "Pall" : [],
                    "Pk"   : [],
                    "unit" : [],
                    "bwm"  : []}
    for i in xrange(tries):
        print "***** loop", i,
        test.random_initialisation() # random from the first round
        for weighting in ["Pall", "Pk", "unit"]:
            test_temp = deepcopy(test) # so we run all of them on the same model
            test_temp.ensemble_averaging(data, weighting, max_iter, verbose = False)
	    try:
                test_temp._cap(norm = True, check = True)
            except Exception:
                print "failed for", weighting
                print repr(test_temp)
		print data
		print "=========== skip"
		next
            test_temp.normalise_hmm_by_reachability(blind = True) ## important (for state-symmetric results)
            error_vec = frobenius_error(reference, test_temp)
            error_sca = arithmetic_mean(error_vec)
            if error_sca < error_sca_min:
                error_sca_min = error_sca # error scalar
                error_vec_min = error_vec # error vector
                best = deepcopy(test_temp)
		print "best so far", error_sca, error_vec_min, repr(best)
            errors_mean[weighting].append(error_sca)
            model_likelihood = _model_likelihood(test_temp, data)
            likelihood_list.append(model_likelihood)
            if model_likelihood > most_likely_measure:
                most_likely_measure = model_likelihood
                most_likely_model   = deepcopy(test_temp)
		print "most likely so far", most_likely_measure, repr(most_likely_model), error_vec, error_sca
    for weighting in ["Pall", "Pk", "unit"]:
        print "-+-+-+-+-+ ensemble %s " % weighting
        print arithmetic_mean(errors_mean[weighting]), sorted(errors_mean[weighting])
    print " * * * * * BEST by supervised comparison * * * * * ", error_vec_min, error_sca_min
    print best, "likelihood:", _model_likelihood(best, data), repr(best)
    print " * * * * * BEST by unsupervised likelihood * * * * * "
    print most_likely_model, "likelihood:", most_likely_measure,"=",_model_likelihood(most_likely_model, data), repr(most_likely_model)
    print "supervised error:", 
    print arithmetic_mean(frobenius_error(reference, most_likely_model)),
    print frobenius_error(reference, most_likely_model)
    print "ref", repr(reference)
    print "average likelihood", arithmetic_mean(likelihood_list), 
    print "likelihood list", likelihood_list

def deterministic_test_several_observations_with_states(tries=10, max_iter = 2000): 
    print "\n\n\n\n\n##### deterministic_test_several_observations_with_states",
    print "(tries = %d, max_iter = %d): " % (tries, max_iter)
    ref = deterministic_hmm
    reference = HMMS(ref.STATES, ref.OBSERVATIONS, ref.A, ref.B, ref.pi)
    data = [reference.simulate(20, show_hidden=True) for i in xrange(20)] 
           # 20 simulations 20-long each
    Q_ =[] ; O_ =[]
    for i in data:
        Q,O = zip(*i)
        Q_.append(Q) ; O_.append(O)

    test = deepcopy(reference)
    test.equiprobable_initialisation()

    errors_mean_Pall = []
    errors_mean_Pk = []
    errors_mean_Unit = []
    errors_mean_bwm = []
    for i in xrange(tries):
        print "\n***** loop", i
        test.random_initialisation() # random from the first round
        test_temp = deepcopy(test) # so we run all of them on the same model
        test_temp.ensemble_averaging(O_, Q_, "Pall", max_iter, verbose = False)        
        errors_mean_Pall.append(arithmetic_mean(frobenius_error(reference, test_temp)))
        test_temp = deepcopy(test)
        test_temp.ensemble_averaging(O_, Q_, "Pk", max_iter, verbose = False)    
        errors_mean_Pk.append(arithmetic_mean(frobenius_error(reference, test_temp)))
        test_temp = deepcopy(test)
        test_temp.ensemble_averaging(O_, Q_, "unit", max_iter, verbose = False)
        errors_mean_Unit.append(arithmetic_mean(frobenius_error(reference, test_temp)))
        test_temp = deepcopy(test)
        test_temp.baum_welch_multiple(O_, Q_, max_iter, verbose = False)
        errors_mean_bwm.append(arithmetic_mean(frobenius_error(reference, test_temp)))


    print "-+-+-+-+-+ ensemble Pall +-+-+-+-+-"
    print sorted(errors_mean_Pall)
    print "-+-+-+-+-+  ensemble Pk  +-+-+-+-+-"
    print sorted(errors_mean_Pk)
    print "-+-+-+-+-+ ensemble Unit +-+-+-+-+-"
    print sorted(errors_mean_Unit)
    print "-+-+-+-+-+  multiple B-W +-+-+-+-+-"
    print sorted(errors_mean_bwm)

if __name__ == '__main__':
#    simple_baum_welch_test(3, 3, 1000)      # one 1000-long sim
#    multiple_baum_welch_test(3, 3, 100, 10) # 100 sims 10-long each
#    deterministic_test_bw_simple()
#    deterministic_test_bw_multiple()
    print repr(deterministic_hmm)
    for i in xrange(50):
        print "\nrun #",i
        deterministic_test_ensemble(tries = 20)
#    deterministic_test_several_observations_with_states()
#    viterbi_likelihoods_suite()

