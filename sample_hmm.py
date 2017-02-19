import hmm


deterministic_hmm = \
         hmm.HMM(['s1', 's2'], ['o1', 'o2', 'o3'],
                [[0.0, 1.0],
                 [1.0, 0.0]],
                [[0.75, 0.0],
                 [0.0, 0.75],
                 [0.25, 0.25]],
                [0.6, 0.4])

deterministic_2s3o_hmm = hmm.HMM(['s1', 's2'], ['o1', 'o2', 'o3'],
                        [[1.0, 0.0],
                         [0.0, 1.0]],
                        [[0.5, 0.0],   # observation of s1
                         [ .5,  .5],   # confusing observation
                         [0.0, 0.5]], # observation of s2
                        [0.5, 0.5])

deterministic_2s3o_hmmBias = hmm.HMM(['s1', 's2'], ['o1', 'o2', 'o3'],
                        [[1.0, 0.0],
                         [0.0, 1.0]],
                        [[0.75, 0.00],   # observation of s1
                         [0.25, 0.25],   # confusing, unlikely observation
                         [0.00, 0.75]], # observation of s2
                        [0.5, 0.5])



#crazy coin:
# -first toss of a sequence is always heads
# -subsequent tosses are always the other state
#           (heads->tails, tails->heads)
# - it's not hidden (observations match states)

#crazy_coin is deterministic
crazy_coin = hmm.HMM(
                ['heads_s', 'tails_s'], 
                ['heads_o', 'tails_o'],
                [[0., 1.],
                 [1., 0.]],
                [[1., 0.],
                 [0., 1.]],
                [1.,0.] # start from heads always
                )

assert 1 == crazy_coin.get_A('heads_s', 'tails_s')
assert 1 == crazy_coin.get_A('tails_s', 'heads_s')
assert 1 == crazy_coin.get_B('heads_s', 'heads_o')
assert 1 == crazy_coin.get_B('tails_s', 'tails_o')
assert 0 == crazy_coin.get_B('heads_s', 'tails_o')
assert 0 == crazy_coin.get_B('tails_s', 'heads_o')
assert 1 == crazy_coin.get_PI('heads_s')
assert 0 == crazy_coin.get_PI('tails_s')

#print "crazy"
#print crazy_coin.simulate(10)
#print crazy_coin.simulate(10, show_hidden = True)

#Fair coin toss:
# - first toss of a sequence 50%-50% heads or tails
# - subsequent tosses have 50% chance of transitioning to the other state
# - in this model I assume s<->o  (de-facto non-hidden Markov chain)
# (in other words, it's not hidden (observations match states))

fair_coin = hmm.HMM(
                ['heads_s', 'tails_s'], 
                ['heads_o', 'tails_o'],
                [[0.5, 0.5],
                 [0.5, 0.5]],
                [[1., 0.],
                 [0., 1.]],
                [0.5, 0.5]
                )

#print "fair"
#for i in xrange(5):
#  print fair_coin.simulate(10)
#  print fair_coin.simulate(10, show_hidden = True)


# HMM hole sample chain:
# - two states, starts equally probably from each (50%-50%)
# - transitions are always to the second state
# - it's not hidden (observations match states)

hmm_hole = hmm.HMM(['transitional_s', 'definitive_s'],
                   ['transitional_o', 'definitive_o'],
                   [[0., 1.],
                    [0., 1.]],
                   [[1., 0.],
                    [0., 1.]],
                   [0.5, 0.5])


# typical weather example (4 states, 3 observations)
#


weather_hmm = hmm.HMMS(['sunny', 'cloudy', 'rainy', 'foggy'], #weather (hidden)
                      ['dry', 'damp', 'wet'], # grass (visible)
                      [[0.5 , 0.3 , 0.1 , 0.1 ],
                       [0.3 , 0.4 , 0.2 , 0.1 ],
                       [0.25, 0.25, 0.3 , 0.2 ],
                       [0.2 , 0.25, 0.25, 0.3 ]],
                      [[0.8 , 0.48, 0.02, 0.02],
                       [0.14, 0.32, 0.26, 0.63],
                       [0.06, 0.20, 0.72, 0.35]],
                      [0.20, 0.50, 0.25, 0.05])

