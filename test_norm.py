from copy import deepcopy
import hmm
from sample_hmm import weather_hmm
from numpy import multiply
from support import frobenius_error, geometric_mean, arithmetic_mean

w1 = deepcopy(weather_hmm)
w2 = deepcopy(weather_hmm)
w3 = deepcopy(weather_hmm)
w4 = deepcopy(weather_hmm)

w2.normalise_hmm_by_A()
w3.normalise_hmm_by_reachability()
w4.normalise_hmm_by_reachability(ignorePI=True)

S = w1.STATES
O = w1.OBSERVATIONS

print "== PI"
for s in S:
  print "w1", s, w1.get_PI(s)
  print "w2", s, w2.get_PI(s)
  print "w3", s, w3.get_PI(s)
  print "w4", s, w4.get_PI(s)
  
print "== A"
for s in S:
  for s2 in S:
    print "w1 A[ %s -> %s ] : %7.4f" % (s, s2, w1.get_A(s,s2))
    print "w2 A[ %s -> %s ] : %7.4f" % (s, s2, w2.get_A(s,s2))
    print "w3 A[ %s -> %s ] : %7.4f" % (s, s2, w3.get_A(s,s2))
    print "w4 A[ %s -> %s ] : %7.4f" % (s, s2, w4.get_A(s,s2))

print "== B"
for s in S:
  for o in O:
    print "w1 B[ %s -> %s ] : %7.4f" % (s, o, w1.get_B(s,o))
    print "w2 B[ %s -> %s ] : %7.4f" % (s, o, w2.get_B(s,o))
    print "w3 B[ %s -> %s ] : %7.4f" % (s, o, w3.get_B(s,o))
    print "w4 B[ %s -> %s ] : %7.4f" % (s, o, w4.get_B(s,o))

print "w1"
print w1
print "w2", frobenius_error(w1,w2), geometric_mean(frobenius_error(w1,w2))
print w2
print "w3", frobenius_error(w1,w3), geometric_mean(frobenius_error(w1,w3))
print w3
print "w4", frobenius_error(w1,w4), geometric_mean(frobenius_error(w1,w4))
print w4

w4.random_initialisation()
print "w4r", frobenius_error(w1,w4), geometric_mean(frobenius_error(w1,w4))

w3.set_A('sunny','cloudy', 0.55)
hmm.normalise(w3.A,1)
print "w3mod", frobenius_error(w1,w3), geometric_mean(frobenius_error(w1,w3)), arithmetic_mean(frobenius_error(w1,w3))

w3.set_A('rainy','cloudy', 0.75) ; hmm.normalise(w3.A,1)
for i in xrange(100): w3.set_A('rainy','cloudy', 0.75) ; hmm.normalise(w3.A,1)

print "w3mod2", frobenius_error(w1,w3), geometric_mean(frobenius_error(w1,w3)), arithmetic_mean(frobenius_error(w1,w3))

w1.normalise_hmm_by_reachability()
w3.normalise_hmm_by_reachability()
print "------w1"
print 
print w1
print "------w3m2"
print 
print w3
print "------w4"
print 
print w4
#w4.reorder_to_match(w1.STATES)
#print w4