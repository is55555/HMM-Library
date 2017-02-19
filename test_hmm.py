import hmm, unittest, os
from numpy import array, multiply, allclose, zeros, take, transpose
from copy import deepcopy

class TestBasicOperation_crazyCoin(unittest.TestCase):
    """Unrealistic coin toss:
          -first toss of a sequence is always heads
          -subsequent tosses are always the other state
                  (heads->tails, tails->heads)
    """
    def setUp(self):
        self.hmm = hmm.HMM(  ['heads_s', 'tails_s'], ['heads_o', 'tails_o'])
        self.hmm2 = hmm.HMMS(['heads_s', 'tails_s'], ['heads_o', 'tails_o'])
        self.hmm2.random_initialisation()
        self.chains_O = []
        self.chains_Q = []
        self.hmm.integrity_check()
        self.hmm2.integrity_check()

    def test_00random_initialisation(self):
        self.hmm.random_initialisation()
        self.hmm.integrity_check()

    def test_01simple_oscillator(self):
        "a 2-state HMM that always changes state in each step"
        multiply(self.hmm.A, 0, self.hmm.A) 
        # reset transition probability matrix
        # (multiply by 0 and clobber on itself)
        self.failUnless(allclose(self.hmm.A, zeros((2, 2), float)))
        self.hmm.set_A('heads_s', 'tails_s', 1 )
        self.hmm.set_A('tails_s', 'heads_s', 1 )
        self.hmm.integrity_check()
        assert 1 == self.hmm.get_A('heads_s', 'tails_s')
        assert 1 == self.hmm.get_A('tails_s', 'heads_s')

    def test_02simple_markov_chain(self):
        """make the HMM equivalent to a simple Markov chain.
        States are effectively not hidden (1-1 equivalence 
        state-observation)"""
        multiply(self.hmm.B, 0, self.hmm.B)
        # reset observation probability matrix
        # (multiply by 0 and clobber on itself)
        self.failUnless(allclose(self.hmm.B, zeros((2, 2), float)))
        self.hmm.set_B('heads_s', 'heads_o', 1)
        self.hmm.set_B('tails_s', 'tails_o', 1)
        self.hmm.integrity_check()
        assert 1 == self.hmm.get_B('heads_s', 'heads_o')
        assert 1 == self.hmm.get_B('tails_s', 'tails_o')
        assert 0 == self.hmm.get_B('heads_s', 'tails_o')
        assert 0 == self.hmm.get_B('tails_s', 'heads_o')

    def test_03always_start_at_s1(self):
        multiply(self.hmm.pi, 0, self.hmm.pi)
        # reset initial probability matrix (pi)
        # (multiply by 0 and clobber on itself)
        self.failUnless(allclose(self.hmm.pi, zeros(2, float)))
        self.hmm.set_PI('heads_s', 1)
        self.hmm.set_PI('tails_s', 0)
        self.hmm.integrity_check()
        assert 1 == self.hmm.get_PI('heads_s')
        assert 0 == self.hmm.get_PI('tails_s')

    def test_04baum_welch_on_simulations(self):
        multiply(self.hmm.A, 0, self.hmm.A) 
        multiply(self.hmm.B, 0, self.hmm.B) 
        multiply(self.hmm.pi, 0, self.hmm.pi) 
        self.hmm.set_A('heads_s', 'tails_s', 1 )
        self.hmm.set_A('tails_s', 'heads_s', 1 )
        self.hmm.set_B('heads_s', 'heads_o', 1)
        self.hmm.set_B('tails_s', 'tails_o', 1)
        self.hmm.set_PI('heads_s', 1)
        self.hmm.set_PI('tails_s', 0)

        for i in xrange(10): # 10 simulations
            sim100 = self.hmm.simulate(100, True)
            Q,O = zip(*sim100)
            self.chains_O.append(O)
            self.chains_Q.append(Q)
        counters={}
        for i in Q: counters[i] = counters.get(i, 0) + 1
        for i in O: counters[i] = counters.get(i, 0) + 1 # based on the fact...
        # ... that states and observations don't have any name conflicts in ...
        # ... this particular example.
        print "simulated 100 coin tosses"
        print "counters:", counters
        assert counters['heads_s'] == 50
        assert counters['tails_s'] == 50
        assert counters['heads_o'] == 50
        assert counters['tails_o'] == 50

        self.hmm2.baum_welch_multiple(self.chains_O, self.chains_Q)
        self.hmm2.integrity_check()
        print "generated HMM on 10 simulations - 100 coin tosses"
        print self.hmm2

class TestBasicOperation_fairCoin(unittest.TestCase):
    """Fair coin toss:
          - first toss of a sequence 50%-50% heads or tails
          - subsequent tosses have 50% chance of transitioning to the other state
          - in this model I assume s<->o  (de-facto non-hidden Markov chain)
    """
    def setUp(self):
        self.hmm = hmm.HMM(  ['heads_s', 'tails_s'], ['heads_o', 'tails_o'])
        self.hmm2 = hmm.HMMS(['heads_s', 'tails_s'], ['heads_o', 'tails_o'])
        self.hmm2.random_initialisation()
        self.chains_O = []
        self.chains_Q = []
        self.hmm.integrity_check()
        self.hmm2.integrity_check()

    def test_00baum_welch_on_simulations(self):
        multiply(self.hmm.A, 0, self.hmm.A) 
        multiply(self.hmm.B, 0, self.hmm.B) 
        multiply(self.hmm.pi, 0, self.hmm.pi) 
        self.hmm.set_A('heads_s', 'tails_s', .5)
        self.hmm.set_A('tails_s', 'heads_s', .5)
        self.hmm.set_A('heads_s', 'heads_s', .5)
        self.hmm.set_A('tails_s', 'tails_s', .5)
        self.hmm.set_B('heads_s', 'heads_o', 1)
        self.hmm.set_B('tails_s', 'tails_o', 1)
        self.hmm.set_PI('heads_s', .5)
        self.hmm.set_PI('tails_s', .5)

        for i in xrange(50): # 50 simulations
            sim100 = self.hmm.simulate(50, True) # 20 tosses each sim
            Q,O = zip(*sim100)
            self.chains_O.append(O)
            self.chains_Q.append(Q)
        
        self.hmm2.baum_welch_multiple(self.chains_O, self.chains_Q)
        self.hmm2.integrity_check()
        print "generated HMM on 50 simulations * 50 coin tosses"
        print self.hmm2


class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.hmm_hole = hmm.HMM(['transitional_s', 'definitive_s'],
                                ['transitional_o', 'definitive_o'],
                        array([[0., 1.],
                               [0., 1.]]),
                        array([[1., 0.],
                               [0., 1.]] ),
                        array([0.5, 0.5]))
        self.hmm_default = hmm.HMM(['s1', 's2'], ['o1', 'o2', 'o3'])
        self.hmm1 = hmm.HMM(['s1', 's2'], ['o1', 'o2'],
                        array([[0.2, 0.8],
                               [0.3, 0.7]]),
                        array( [[1., 0.2],
                                [0., 0.8]] ),
                        array([0.3, 0.7]))
        self.hmm2 = hmm.HMM(['s1', 's2'], ['o1', 'o2', 'o3'],
                        array([[1.0, 0.0],
                               [0.0, 1.0]]),
                        array([[0.5, 0.0],
                               [ .5,  .5],
                               [0.0, 0.5]]),
                        array([0.5, 0.5]))
    
    def test_alpha_scaled(self):
        Bo = take(self.hmm_hole.B, [0, 1], 0)
        alpha, scaling_factors = self.hmm_hole.alpha_scaled(self.hmm_hole.A, Bo, self.hmm_hole.pi )
        self.failUnless(allclose(
                alpha, 
                array([[1, 0], [0, 1]])   ))
        self.failUnless(allclose(
                scaling_factors,
                array([2., 1.])   ))

        Bo = take(self.hmm_default.B, [0, 1, 2], 0)
        alpha, scaling_factors = self.hmm_default.alpha_scaled(self.hmm_default.A, Bo, self.hmm_default.pi )
        self.failUnless(allclose(
                alpha, 
                array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])    ))
        self.failUnless(allclose(
                scaling_factors,
                array([3., 3., 3.])    ))

    def test_beta_scaled(self):        
        Bo = take(self.hmm_hole.B, [0, 1, 1], 0)
        beta = self.hmm_hole.beta_scaled(self.hmm_hole.A, Bo, array([2., 1., 1.]))
        self.failUnless(allclose(
                beta, 
                array([[2., 2.], [1., 1.], [1., 1.]])    ))

        Bo = take(self.hmm_default.B, [0, 1, 2], 0)
        beta = self.hmm_default.beta_scaled(self.hmm_default.A, Bo, array([3., 3., 3.]))
        self.failUnless(allclose(
                beta, 
                array([[3., 3.], [3., 3.], [3., 3.]])    ))

    def test_gamma(self):
        A, B, PI = self.hmm_hole.A, self.hmm_hole.B, self.hmm_hole.pi
        Bo = take(B, [0, 1, 1], 0)
        alpha, scale_factors = self.hmm_hole.alpha_scaled(A, Bo, PI )
        beta = self.hmm_hole.beta_scaled( A, Bo, scale_factors )
        gamma = self.hmm_hole._bw_gamma(alpha, beta, scale_factors)
        self.failUnless(allclose(
                gamma,
                array([[1, 0], [0, 1], [0, 1]], float)    ))

        A, B, PI = self.hmm_default.A, self.hmm_default.B, self.hmm_default.pi
        Bo = take(B, [0, 1, 2], 0)
        alpha, scale_factors = self.hmm_default.alpha_scaled(A, Bo, PI )
        beta = self.hmm_default.beta_scaled( A, Bo, scale_factors )
        gamma = self.hmm_default._bw_gamma(alpha, beta, scale_factors)
        self.failUnless(allclose(
                gamma, 
                array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])    ))

    def test_ksi(self):
        Bo = take(self.hmm_hole.B, [0, 1, 1], 0)
        alpha, scale_factors = self.hmm_hole.alpha_scaled(self.hmm_hole.A, 
                                                Bo, self.hmm_hole.pi)
        beta = self.hmm_hole.beta_scaled(self.hmm_hole.A, Bo, scale_factors)
        ksy = self.hmm_hole.ksi( self.hmm_hole.A, Bo, alpha, beta )
        self.failUnless(allclose(
                ksy, 
                array([ [[0., 1.], [0., 0.]], 
                        [[0., 0.], [0., 1.]]])    ))


        Bo = take(self.hmm_default.B, [0, 1, 2], 0)
        alpha, scale_factors = self.hmm_default.alpha_scaled(self.hmm_default.A, 
                                                Bo, self.hmm_default.pi)
        beta = self.hmm_default.beta_scaled(self.hmm_default.A, Bo, scale_factors)
        ksy = self.hmm_default.ksi(self.hmm_default.A, Bo, alpha, beta)
        self.failUnless(allclose(
                ksy, 
                array([ [[0.25, 0.25], [0.25, 0.25]], 
                        [[0.25, 0.25], [0.25, 0.25]]])    ))

    def test_bw_converged(self):
        self.hmm_hole._bw_converged(self.hmm_hole.A, self.hmm_hole.B, self.hmm_hole.pi)
     
    def test_bw_model_recalculation(self):
        gamma = array([[1., 0.], [0., 1.], [0., 1]])
        ksi = array([[[0., 1.], [0., 0.]], [[0., 0.], [0., 1.]] ])
        Abar = array([[0., 1.], [0., 1.]])
        Bbar = array([[1., 0.], [0., 1.]])
        pibar = array([1., 0.])
        A, B, PI = self.hmm_hole._bw_model_recalculation(gamma, ksi, [0, 1, 1])
        self.failUnless(allclose(Abar, A))
        self.failUnless(allclose(Bbar, B))
        self.failUnless(allclose(pibar, PI))

    def test_simulate(self):
        sim = self.hmm1.simulate(10)
        self.assertEquals(len(sim), 10)
        for element in sim: self.failUnless(element in self.hmm1.OBSERVATIONS)
        sim = self.hmm1.simulate(10, show_hidden=True)
        self.assertEquals(len(sim), 10)
        for pair in sim:
            self.failUnless(pair[0] in self.hmm1.STATES)
            self.failUnless(pair[1] in self.hmm1.OBSERVATIONS)

    def test_normalise_hmm_by_A(self):
        res_a = array([[0.7, 0.3],
                       [0.8, 0.2]])
        res_b = array([[0.2, 1.],
                       [0.8, 0.]])
        res_pi = array([0.7, 0.3])
        A, B = zeros( (self.hmm1.N, self.hmm1.N), float)
        PI = zeros( self.hmm1.N)
        #A, B, PI = self.hmm1.A_B_PI()
        #print "\n", A, "\n", B, "\n", PI
        A, B, PI = self.hmm1.normalise_hmm_by_A()
        #print "\n", A, "\n", B, "\n", PI
        self.failUnless(allclose(A, res_a))
        self.failUnless(allclose(B, res_b))
        self.failUnless(allclose(PI, res_pi))
        
    def test_reorder(self):
        res_a = array([[0.7, 0.3], [0.8, 0.2]])
        res_b = array([[0.2, 1.], [0.8, 0.]])
        res_pi = array([0.7, 0.3])
        A, B = zeros( (self.hmm1.N, self.hmm1.N), float)
        PI = zeros( self.hmm1.N)
        P = array([1, 0])
        A, B, PI = self.hmm1.reorder(P)
        self.failUnless(allclose(A, res_a))
        self.failUnless(allclose(B, res_b))
        self.failUnless(allclose(PI, res_pi))

    def test_correct_matrix( self):
        A      = array([[0., 0.],
                        [0.2, 0.8]])
        result = array([[0.5, 0.5],
                        [0.2, 0.8]])
        MM = hmm._bwm_correct_matrix(A, 1, .5)
        self.failUnless(allclose(result, MM))
        A      = array([[1., 0.],
                        [0., 0.]])
        result = array([[1., 0.],
                        [0.5, 0.5]])
        MM = hmm._bwm_correct_matrix(A, 1, .5)
        self.failUnless( allclose(result, MM))
        A      = array([[0., 0.],
                        [0., 0.]])
        result = array([[0.5, 0.5],
                        [0.5, 0.5]])
        MM = hmm._bwm_correct_matrix(A, 1, .5)
        self.failUnless(allclose(result, MM))
        B      = array([[0., 0.2],
                        [0., 0.8]])
        result = array([[0.5, 0.2],
                        [0.5, 0.8]])
        MM = hmm._bwm_correct_matrix(B, 0, .5)
        self.failUnless(allclose(result, MM))
        B      = array([[1., 0.],
                        [0., 0.]])
        result = array([[1., 0.5],
                        [0., 0.5]])
        MM = hmm._bwm_correct_matrix(B, 0, .5)
        self.failUnless( allclose(result, MM))
        B      = array([[0., 0.],
                        [0., 0.]])
        result = array([[0.5, 0.5],
                        [0.5, 0.5]])
        MM = hmm._bwm_correct_matrix(B, 0, .5)
        self.failUnless(allclose(result, MM))



class TestStates(unittest.TestCase):
    def setUp(self):
        self.faircoin = hmm.HMMS(['heads_s', 'tails_s'], ['heads_o', 'tails_o'],
                        array([[0.5, 0.5],
                               [0.5, 0.5]]),
                        array([[1., 0.],
                               [0., 1.]]),
                        array([0.5, 0.5]))

        self.weatherhmm = hmm.HMMS(['sunny', 'cloudy', 'rainy', 'foggy'], #weather (hidden)
                        ['dry', 'damp', 'wet'], # grass (visible)
                            array([[0.5 , 0.3 , 0.1 , 0.1 ],
                                   [0.3 , 0.4 , 0.2 , 0.1 ],
                                   [0.25, 0.25, 0.3 , 0.2 ],
                                   [0.2 , 0.25, 0.25, 0.3 ] ]),
#                           array([[0.6 , 0.38, 0.01, 0.01], # normalised to the wrong axis
#                                   [0.1 , 0.25, 0.20, 0.45],
#                                   [0.05, 0.15, 0.55, 0.25]
#                                   ]),
                            array([[ 0.8 ,  0.48,  0.02,  0.02],
                                   [ 0.14,  0.32,  0.26,  0.63],
                                   [ 0.06,  0.20,  0.72,  0.35]]),

                            array([0.20, 0.50, 0.25, 0.05]))
        self.sampleQ = ['cloudy', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'cloudy',
                        'cloudy', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 
                        'sunny', 'cloudy', 'foggy', 'sunny', 'sunny', 'rainy', 'rainy', 'foggy', 
                        'foggy', 'foggy', 'sunny', 'sunny', 'sunny', 'sunny', 'cloudy', 'cloudy', 
                        'sunny', 'foggy', 'foggy', 'rainy', 'sunny', 'cloudy', 'cloudy', 'sunny', 
                        'foggy', 'rainy', 'sunny', 'sunny', 'foggy', 'rainy', 'rainy', 'cloudy', 
                        'cloudy', 'cloudy', 'rainy', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 
                        'sunny', 'sunny', 'sunny', 'foggy', 'rainy', 'foggy', 'cloudy', 'sunny', 
                        'sunny', 'cloudy', 'foggy', 'sunny', 'sunny', 'sunny', 'sunny', 'foggy', 
                        'rainy', 'rainy', 'sunny', 'cloudy', 'cloudy', 'sunny', 'sunny', 'sunny', 
                        'sunny', 'cloudy', 'sunny', 'sunny', 'foggy', 'sunny', 'sunny', 'foggy', 
                        'cloudy', 'cloudy', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 
                        'rainy', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'rainy', 'sunny', 
                        'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'cloudy', 'cloudy', 
                        'cloudy', 'cloudy', 'sunny', 'sunny', 'rainy', 'cloudy', 'cloudy', 'cloudy', 
                        'cloudy', 'sunny', 'sunny', 'cloudy', 'foggy', 'foggy', 'foggy', 'rainy', 
                        'sunny', 'sunny', 'cloudy', 'cloudy', 'rainy', 'rainy', 'cloudy', 'cloudy',
                        'cloudy', 'cloudy', 'foggy', 'foggy', 'rainy', 'sunny', 'sunny', 'sunny', 
                        'cloudy', 'cloudy', 'sunny', 'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 
                        'sunny', 'foggy', 'rainy', 'rainy', 'rainy', 'rainy', 'cloudy', 'cloudy', 
                        'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'rainy', 'rainy', 'cloudy', 
                        'rainy', 'rainy', 'foggy', 'cloudy', 'cloudy', 'sunny', 'cloudy', 'cloudy', 
                        'rainy', 'sunny', 'sunny', 'sunny', 'cloudy', 'foggy', 'foggy', 'cloudy', 
                        'sunny', 'sunny', 'sunny', 'rainy', 'cloudy', 'sunny', 'rainy', 'rainy', 
                        'foggy', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'cloudy', 'cloudy', 
                        'rainy', 'cloudy', 'sunny', 'foggy', 'cloudy', 'foggy', 'rainy', 'sunny', 
                        'sunny', 'rainy', 'foggy', 'sunny', 'sunny', 'foggy', 'cloudy', 'sunny', 
                        'sunny', 'foggy', 'sunny', 'foggy', 'cloudy', 'cloudy', 'sunny', 'sunny', 
                        'sunny', 'cloudy', 'cloudy', 'sunny', 'cloudy', 'foggy', 'foggy', 'foggy', 
                        'foggy', 'foggy', 'foggy', 'cloudy', 'foggy', 'rainy', 'sunny', 'rainy', 
                        'cloudy', 'cloudy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'sunny', 'foggy', 
                        'rainy', 'rainy', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'cloudy', 
                        'sunny', 'sunny', 'sunny', 'cloudy', 'cloudy', 'sunny', 'sunny', 'sunny', 
                        'cloudy', 'sunny', 'sunny', 'foggy', 'foggy', 'foggy', 'cloudy', 'sunny', 
                        'cloudy', 'sunny', 'rainy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'foggy', 
                        'sunny', 'rainy', 'sunny', 'sunny', 'sunny', 'cloudy', 'sunny', 'sunny', 
                        'cloudy', 'sunny', 'sunny', 'sunny', 'sunny', 'foggy', 'sunny', 'sunny', 
                        'sunny', 'sunny', 'sunny', 'sunny', 'cloudy', 'cloudy', 'sunny', 'foggy', 
                        'foggy', 'foggy', 'rainy', 'foggy', 'cloudy', 'cloudy', 'cloudy', 'sunny', 
                        'sunny', 'foggy', 'sunny', 'rainy', 'foggy', 'sunny', 'sunny', 'rainy', 
                        'foggy', 'rainy', 'foggy', 'rainy', 'cloudy', 'cloudy', 'sunny', 'sunny', 
                        'sunny', 'cloudy', 'foggy', 'cloudy', 'sunny', 'sunny', 'sunny', 'sunny', 
                        'rainy', 'sunny', 'sunny', 'sunny', 'cloudy', 'cloudy', 'sunny', 'sunny', 
                        'sunny', 'rainy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'rainy', 'rainy', 
                        'rainy', 'rainy', 'rainy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 
                        'sunny', 'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 'sunny', 'cloudy', 
                        'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 'cloudy', 'sunny', 'cloudy', 
                        'sunny', 'cloudy', 'rainy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'sunny', 
                        'sunny', 'cloudy', 'foggy', 'cloudy', 'cloudy', 'foggy', 'cloudy', 'sunny', 
                        'sunny', 'sunny', 'cloudy', 'sunny', 'sunny', 'rainy', 'cloudy', 'cloudy', 
                        'rainy', 'foggy', 'cloudy', 'rainy', 'foggy', 'cloudy', 'rainy', 'cloudy', 
        'rainy', 'rainy', 'foggy', 'rainy', 'foggy', 'foggy', 'foggy', 'sunny', 'cloudy', 'foggy', 
        'sunny', 'cloudy', 'cloudy', 'sunny', 'sunny', 'sunny', 'foggy', 'cloudy', 'cloudy', 'rainy', 
        'foggy', 'foggy', 'foggy', 'foggy', 'sunny', 'sunny', 'rainy', 'rainy', 'rainy', 'foggy', 
        'rainy', 'foggy', 'sunny', 'rainy', 'cloudy', 'rainy', 'sunny', 'sunny', 'cloudy', 'sunny', 
        'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 'cloudy', 'sunny', 'sunny', 'cloudy', 'foggy', 
        'cloudy', 'cloudy', 'rainy', 'foggy', 'rainy', 'cloudy', 'sunny', 'sunny', 'rainy', 'rainy', 
        'rainy', 'foggy', 'foggy', 'sunny', 'sunny', 'cloudy', 'foggy', 'foggy', 'rainy', 'foggy',
        'foggy', 'cloudy', 'cloudy', 'sunny', 'sunny', 'cloudy', 'foggy', 'cloudy', 'sunny', 'foggy',
        'foggy', 'rainy', 'cloudy', 'cloudy', 'cloudy', 'rainy', 'cloudy', 'foggy', 'cloudy', 'sunny',
        'sunny', 'sunny', 'cloudy', 'foggy', 'foggy', 'foggy', 'cloudy', 'sunny', 'foggy', 'cloudy', 
        'cloudy', 'sunny', 'sunny', 'sunny', 'foggy', 'rainy', 'cloudy', 'cloudy', 'foggy', 'foggy',
        'foggy', 'cloudy', 'cloudy', 'rainy', 'sunny', 'sunny', 'sunny', 'cloudy', 'foggy', 'sunny',
        'sunny', 'sunny', 'sunny', 'sunny', 'rainy', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny',
        'rainy', 'foggy', 'sunny', 'sunny', 'sunny', 'cloudy', 'sunny', 'sunny', 'foggy', 'sunny',
        'foggy', 'cloudy', 'sunny', 'rainy', 'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 'rainy',
        'rainy', 'foggy', 'foggy', 'rainy', 'foggy', 'cloudy', 'rainy', 'foggy', 'rainy', 'cloudy',
        'foggy', 'foggy', 'rainy', 'cloudy', 'cloudy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'rainy',
        'foggy', 'foggy', 'foggy', 'rainy', 'rainy', 'rainy', 'sunny', 'sunny', 'sunny', 'sunny',
        'sunny', 'sunny', 'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'rainy', 'sunny', 'sunny',
        'sunny', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 
        'cloudy', 'foggy', 'sunny', 'cloudy', 'rainy', 'rainy', 'rainy', 'sunny', 'cloudy', 'cloudy',
        'cloudy', 'cloudy', 'rainy', 'foggy', 'foggy', 'rainy', 'foggy', 'sunny', 'cloudy', 'cloudy',
        'sunny', 'rainy', 'sunny', 'sunny', 'sunny', 'cloudy', 'cloudy', 'sunny', 'cloudy', 'cloudy',
        'rainy', 'cloudy', 'rainy', 'sunny', 'foggy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'sunny',
        'sunny', 'sunny', 'cloudy', 'foggy', 'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 'sunny',
        'cloudy', 'rainy', 'rainy', 'sunny', 'sunny', 'sunny', 'cloudy', 'rainy', 'rainy', 'sunny',
        'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'cloudy', 'rainy', 'rainy',
        'cloudy', 'cloudy', 'rainy', 'foggy', 'rainy', 'sunny', 'sunny', 'cloudy', 'rainy', 'sunny',
        'sunny', 'foggy', 'rainy', 'sunny', 'foggy', 'cloudy', 'sunny', 'sunny', 'sunny', 'foggy',
        'foggy', 'foggy', 'sunny', 'cloudy', 'foggy', 'rainy', 'rainy', 'foggy', 'cloudy', 'sunny',
        'cloudy', 'rainy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'rainy', 'cloudy', 'sunny', 
        'cloudy', 'sunny', 'cloudy', 'sunny', 'foggy', 'sunny', 'cloudy', 'sunny', 'cloudy', 'sunny',
        'sunny', 'sunny', 'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 'cloudy', 'foggy', 'rainy',
        'sunny', 'sunny', 'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny',
        'sunny', 'sunny', 'sunny', 'foggy', 'rainy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'sunny',
        'sunny', 'cloudy', 'rainy', 'sunny', 'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 'sunny',
        'cloudy', 'foggy', 'sunny', 'cloudy', 'cloudy', 'rainy', 'sunny', 'cloudy', 'cloudy', 'sunny',
        'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'cloudy', 'rainy', 'cloudy', 'sunny',
        'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'sunny', 'sunny', 'foggy', 'foggy',
        'rainy', 'sunny', 'foggy', 'rainy', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'sunny',
        'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'cloudy',
        'rainy', 'rainy', 'rainy', 'rainy', 'cloudy', 'cloudy', 'sunny', 'foggy', 'foggy', 'sunny',
        'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'rainy', 'rainy', 'sunny', 
        'cloudy', 'sunny', 'sunny', 'foggy', 'rainy', 'rainy', 'cloudy', 'rainy', 'foggy', 'sunny', 
        'cloudy', 'cloudy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'sunny', 'rainy', 'foggy', 'cloudy',
        'rainy', 'foggy', 'rainy', 'cloudy', 'sunny', 'cloudy', 'rainy', 'foggy', 'sunny', 'foggy',
        'foggy', 'rainy', 'cloudy', 'foggy', 'sunny', 'cloudy', 'rainy', 'sunny', 'sunny', 'cloudy',
        'foggy', 'sunny', 'sunny', 'foggy', 'rainy', 'foggy', 'foggy', 'rainy', 'foggy', 'cloudy',
        'sunny', 'cloudy', 'sunny', 'sunny', 'sunny', 'rainy', 'cloudy', 'cloudy', 'foggy', 'cloudy',
        'rainy', 'foggy', 'rainy', 'rainy', 'sunny', 'cloudy', 'rainy', 'rainy', 'sunny', 'foggy',
        'foggy', 'cloudy', 'sunny', 'cloudy', 'rainy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'foggy',
        'rainy', 'sunny', 'sunny', 'sunny', 'sunny', 'cloudy', 'sunny', 'sunny', 'foggy', 'sunny',
        'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny',
        'cloudy', 'cloudy', 'cloudy', 'cloudy', 'rainy', 'foggy', 'rainy', 'foggy', 'foggy', 'foggy',
        'cloudy', 'rainy', 'sunny', 'cloudy', 'cloudy', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'sunny',
        'sunny', 'cloudy', 'cloudy', 'sunny', 'sunny', 'rainy', 'foggy', 'sunny', 'sunny', 'sunny',
        'sunny', 'cloudy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'rainy',
        'sunny', 'sunny', 'sunny', 'sunny', 'foggy', 'cloudy', 'sunny', 'cloudy', 'foggy', 'cloudy',
        'cloudy', 'cloudy', 'cloudy', 'sunny', 'cloudy', 'sunny', 'sunny', 'cloudy', 'foggy', 'foggy',
        'sunny', 'sunny', 'sunny']

        self.sampleMultipleQ = [('cloudy', 'cloudy', 'cloudy', 'rainy', 'rainy', 'cloudy', 'sunny', 'sunny', 'sunny', 'cloudy'), ('foggy', 'rainy', 'rainy', 'rainy', 'cloudy', 'foggy', 'foggy', 'foggy', 'foggy', 'foggy'), ('rainy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'rainy', 'rainy', 'rainy', 'cloudy', 'cloudy'), ('rainy', 'foggy', 'cloudy', 'rainy', 'foggy', 'cloudy', 'cloudy', 'cloudy', 'foggy', 'cloudy'), ('cloudy', 'rainy', 'rainy', 'rainy', 'rainy', 'rainy', 'cloudy', 'sunny', 'sunny', 'sunny'), ('sunny', 'sunny', 'sunny', 'cloudy', 'rainy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy'), ('sunny', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'foggy', 'cloudy'), ('cloudy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'rainy', 'sunny', 'sunny'), ('cloudy', 'cloudy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'cloudy'), ('cloudy', 'sunny', 'cloudy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'cloudy'), ('cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'rainy', 'rainy', 'cloudy'), ('cloudy', 'foggy', 'sunny', 'rainy', 'foggy', 'rainy', 'rainy', 'rainy', 'rainy', 'cloudy'), ('cloudy', 'cloudy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'sunny', 'sunny', 'sunny', 'cloudy'), ('cloudy', 'foggy', 'rainy', 'cloudy', 'cloudy', 'sunny', 'sunny', 'foggy', 'sunny', 'cloudy'), ('cloudy', 'sunny', 'cloudy', 'cloudy', 'rainy', 'sunny', 'rainy', 'foggy', 'rainy', 'cloudy'), ('cloudy', 'sunny', 'cloudy', 'rainy', 'foggy', 'cloudy', 'sunny', 'foggy', 'rainy', 'sunny'), ('sunny', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'cloudy', 'sunny', 'cloudy', 'sunny', 'rainy'), ('cloudy', 'rainy', 'rainy', 'rainy', 'rainy', 'cloudy', 'sunny', 'sunny', 'foggy', 'sunny'), ('cloudy', 'sunny', 'cloudy', 'sunny', 'cloudy', 'sunny', 'cloudy', 'rainy', 'cloudy', 'rainy'), ('cloudy', 'rainy', 'rainy', 'foggy', 'rainy', 'rainy', 'foggy', 'foggy', 'cloudy', 'cloudy')]

        self.sampleMultipleSim = [
[('cloudy', 'damp'), ('cloudy', 'wet'), ('rainy', 'wet'), ('sunny', 'dry'), ('sunny', 'dry')],
[('sunny', 'dry'), ('sunny', 'dry'), ('cloudy', 'dry'), ('sunny', 'dry'), ('cloudy', 'dry')], [('cloudy', 'dry'), ('foggy', 'wet'), ('rainy', 'wet'), ('sunny', 'dry'), ('cloudy', 'dry')], [('cloudy', 'dry'), ('cloudy', 'dry'), ('rainy', 'damp'), ('cloudy', 'damp'), ('sunny', 'dry')],
[('rainy', 'wet'), ('sunny', 'dry'), ('sunny', 'wet'), ('sunny', 'damp'), ('foggy', 'damp')], [('cloudy', 'dry'), ('rainy', 'wet'), ('sunny', 'dry'), ('cloudy', 'wet'), ('sunny', 'damp')], [('cloudy', 'wet'), ('rainy', 'wet'), ('rainy', 'damp'), ('foggy', 'damp'), ('foggy', 'wet')], [('rainy', 'damp'), ('cloudy', 'damp'), ('cloudy', 'dry'), ('rainy', 'damp'), ('sunny', 'dry')], [('cloudy', 'dry'), ('cloudy', 'wet'), ('sunny', 'dry'), ('rainy', 'wet'), ('rainy', 'wet')], [('cloudy', 'dry'), ('rainy', 'wet'), ('rainy', 'wet'), ('sunny', 'damp'), ('cloudy', 'dry')], [('rainy', 'wet'), ('sunny', 'damp'), ('sunny', 'damp'), ('rainy', 'wet'), ('cloudy', 'damp')], [('cloudy', 'dry'), ('cloudy', 'dry'), ('sunny', 'dry'), ('sunny', 'dry'), ('rainy', 'wet')], [('sunny', 'dry'), ('sunny', 'dry'), ('cloudy', 'wet'), ('sunny', 'dry'), ('sunny', 'dry')], [('cloudy', 'dry'), ('cloudy', 'wet'), ('cloudy', 'dry'), ('cloudy', 'dry'), ('foggy', 'damp')], [('rainy', 'dry'), ('sunny', 'dry'), ('cloudy', 'wet'), ('foggy', 'damp'), ('rainy', 'wet')],
[('cloudy', 'wet'), ('cloudy', 'damp'), ('cloudy', 'dry'), ('foggy', 'damp'), ('rainy', 'wet')], [('cloudy', 'dry'), ('cloudy', 'dry'), ('sunny', 'wet'), ('sunny', 'dry'), ('cloudy', 'damp')], [('sunny', 'damp'), ('sunny', 'dry'), ('foggy', 'damp'), ('rainy', 'wet'), ('foggy', 'damp')], [('sunny', 'wet'), ('cloudy', 'dry'), ('cloudy', 'dry'), ('sunny', 'dry'), ('sunny', 'dry')], [('cloudy', 'damp'), ('cloudy', 'damp'), ('rainy', 'wet'), ('cloudy', 'dry'), ('sunny', 'dry')], [('cloudy', 'dry'), ('sunny', 'dry'), ('sunny', 'dry'), ('cloudy', 'dry'), ('sunny', 'dry')], [('sunny', 'damp'), ('foggy', 'damp'), ('cloudy', 'dry'), ('cloudy', 'damp'), ('sunny', 'dry')], [('cloudy', 'dry'), ('sunny', 'dry'), ('sunny', 'dry'), ('sunny', 'dry'), ('rainy', 'wet')], [('sunny', 'dry'), ('sunny', 'dry'), ('cloudy', 'dry'), ('sunny', 'dry'), ('sunny', 'dry')], [('sunny', 'dry'), ('sunny', 'dry'), ('cloudy', 'wet'), ('foggy', 'damp'), ('cloudy', 'wet')], [('cloudy', 'dry'), ('sunny', 'dry'), ('sunny', 'dry'), ('cloudy', 'damp'), ('cloudy', 'damp')], [('cloudy', 'dry'), ('cloudy', 'damp'), ('rainy', 'wet'), ('cloudy', 'dry'), ('cloudy', 'dry')], [('sunny', 'dry'), ('foggy', 'wet'), ('sunny', 'dry'), ('cloudy', 'wet'), ('cloudy', 'damp')], [('sunny', 'dry'), ('sunny', 'dry'), ('cloudy', 'dry'), ('sunny', 'dry'), ('sunny', 'dry')],
[('sunny', 'damp'), ('foggy', 'damp'), ('foggy', 'damp'), ('cloudy', 'damp'), ('cloudy', 'wet')], [('rainy', 'wet'), ('rainy', 'wet'), ('foggy', 'wet'), ('cloudy', 'wet'), ('cloudy', 'dry')], [('rainy', 'wet'), ('rainy', 'wet'), ('sunny', 'damp'), ('sunny', 'dry'), ('cloudy', 'dry')], [('cloudy', 'wet'), ('sunny', 'dry'), ('sunny', 'dry'), ('sunny', 'dry'), ('rainy', 'damp')], [('rainy', 'wet'), ('sunny', 'wet'), ('rainy', 'wet'), ('cloudy', 'damp'), ('sunny', 'dry')], [('cloudy', 'wet'), ('sunny', 'dry'), ('sunny', 'dry'), ('sunny', 'wet'), ('sunny', 'damp')], [('sunny', 'dry'), ('cloudy', 'wet'), ('foggy', 'damp'), ('foggy', 'damp'), ('foggy', 'damp')], [('rainy', 'wet'), ('rainy', 'damp'), ('cloudy', 'dry'), ('cloudy', 'damp'), ('sunny', 'dry')], [('cloudy', 'dry'), ('sunny', 'wet'), ('cloudy', 'wet'), ('sunny', 'dry'), ('rainy', 'damp')], [('rainy', 'damp'), ('rainy', 'wet'), ('cloudy', 'dry'), ('sunny', 'dry'), ('cloudy', 'damp')], [('sunny', 'dry'), ('sunny', 'dry'), ('sunny', 'dry'), ('sunny', 'dry'), ('cloudy', 'damp')]]

    def test_00learn_A(self):
        print self
        sim = self.weatherhmm.simulate(1000, True)
        #print sim
        learntW = deepcopy(self.weatherhmm)
        Q,O = zip(*sim)
        learntW._learn_A(Q)
        #print "learnt", learntW
        #print "orig", self.weatherhmm

        # starting from the original A doesn't matter for _learn_A, it doesn't use
        # previous A values at all. It clobbers them.
        A_learnt_seq = [[ 0.57317073,  0.26829268,  0.06585366,  0.09268293], 
                        [ 0.34256055,  0.39100346,  0.15570934,  0.11072664],
                        [ 0.29054054,  0.20945946,  0.24324324,  0.25675676],
                        [ 0.22368421,  0.22368421,  0.26315789,  0.28947368]]

        learntW = deepcopy(self.weatherhmm)
        learntW._learn_A(self.sampleQ)
        self.failUnless(allclose(A_learnt_seq, learntW.A))

    def test_01baum_welch_multiple_learn_A(self):
        print self
#        for i in xrange(20):
#            sim = self.weatherhmm.simulate(10, True)
#            Q,O = zip(*sim)
#            self.sampleMultipleQ.append(Q)

        self.weatherhmm.equiprobable_initialisation()
        self.weatherhmm._baum_welch_multiple_learn_A(self.sampleMultipleQ)

        result = [[ 0.50083333,  0.32      ,  0.0875    ,  0.09166667], # 20*10 is remarkably 
                  [ 0.27916667,  0.40119048,  0.19047619,  0.12916667], # good compared to
                  [ 0.09166667,  0.29083333,  0.49083333,  0.12666667], # 1*1000 in this case
                  [ 0.1       ,  0.14166667,  0.15166667,  0.60666667]] # it's actually better

        self.failUnless( allclose(result, self.weatherhmm.A))

    def test_02baumwelch(self):
        print self
        #sim = self.weatherhmm.simulate(50, True)
        #Q,O = zip(*sim)
        Q = ('rainy', 'rainy', 'rainy', 'sunny', 'cloudy', 'rainy', 'foggy', 'sunny', 'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'sunny', 'cloudy', 'sunny', 'foggy', 'cloudy', 'rainy', 'rainy', 'sunny', 'sunny', 'rainy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'cloudy', 'foggy', 'cloudy', 'cloudy', 'sunny', 'cloudy', 'sunny', 'cloudy', 'foggy', 'rainy', 'sunny', 'foggy', 'cloudy', 'cloudy', 'cloudy', 'sunny')
        O = ('wet', 'wet', 'wet', 'wet', 'damp', 'wet', 'wet', 'dry', 'dry', 'damp', 'damp', 'damp', 'damp', 'dry', 'wet', 'dry', 'dry', 'damp', 'damp', 'damp', 'damp', 'dry', 'dry', 'damp', 'damp', 'dry', 'dry', 'dry', 'dry', 'wet', 'wet', 'dry', 'dry', 'dry', 'wet', 'damp', 'wet', 'wet', 'damp', 'damp', 'dry', 'dry', 'wet', 'wet', 'dry', 'damp', 'damp', 'damp', 'dry', 'dry')
        A = array([
                [3.076923E-01,  4.615385E-01,  7.692308E-02,  1.538462E-01], 
                [2.173913E-01,  6.086957E-01,  8.695652E-02,  8.695652E-02],
                [5.000000E-01,  0.000000E+00,  3.750000E-01,  1.250000E-01], 
                [2.000000E-01,  6.000000E-01,  2.000000E-01,  0.000000E+00] ])
        B = array([
                [7.457440E-01,  4.229095E-01,  1.066648E-02,  1.712988E-02],
                [1.848188E-01,  4.105313E-01,  2.197753E-01,  6.442566E-01],
                [6.943719E-02,  1.665592E-01,  7.695582E-01,  3.386136E-01] ])
        PI = array([0., 0., 1., 1.301721E-09])
        self.weatherhmm.baum_welch_o(O, Q)
        self.weatherhmm.integrity_check()
        #print
        #print "Q",Q
        #print "O",O
        self.failUnless(allclose(A,  self.weatherhmm.A))
        self.failUnless(allclose(B,  self.weatherhmm.B))
        self.failUnless(allclose(PI, self.weatherhmm.pi))
        self.weatherhmm.equiprobable_initialisation()
        self.weatherhmm.baum_welch_o(O, Q)
        A=array(        [[ 0.30769231,  0.46153846,  0.07692308,  0.15384615],
                         [ 0.2173913 ,  0.60869565,  0.08695652,  0.08695652],
                         [ 0.5       ,  0.        ,  0.375     ,  0.125     ],
                         [ 0.2       ,  0.6       ,  0.2       ,  0.        ]])

        B=array(        [[ 0.37937637,  0.38559553,  0.3698268 ,  0.36954106], # sucky (not enough
                         [ 0.33955344,  0.34487389,  0.3310762 ,  0.33071485], # data to inter so
                         [ 0.28107019,  0.26953058,  0.29909699,  0.2997441 ]])# many variables)

        PI=array([0.25, 0.25, 0.25, 0.25]) # not sure why it didn't learn PI here (2 iterations)

        self.failUnless(allclose(A,  self.weatherhmm.A))
        self.failUnless(allclose(B,  self.weatherhmm.B))
        self.failUnless(allclose(PI, self.weatherhmm.pi))


    def test_03baum_welch_multiple(self):
        print self
        #for i in xrange(40):
        #    sim = self.weatherhmm.simulate(5, True)
        #    self.sampleMultipleSim.append(sim)
        print "\n multiple Baum-Welch on 40*5 (weather example)- this can take a while"
        #print self.sampleMultipleSim
        Q_ =[] ; O_ =[]
        for i in self.sampleMultipleSim:
            Q,O = zip(*i)
            Q_.append(Q) ; O_.append(O)

        self.weatherhmm.baum_welch_multiple(O_, Q_)
        self.weatherhmm.integrity_check()
        A = [[5.145833E-01,  2.979167E-01,  1.041667E-01,  8.333333E-02], 
             [3.916667E-01,  3.812500E-01,  1.125000E-01,  1.145833E-01], 
             [1.875000E-01,  1.375000E-01,  6.250000E-01,  5.000000E-02], 
             [2.500000E-02,  8.750000E-02,  1.000000E-01,  7.875000E-01]]
        B = [[2.026588E-01,  9.955530E-01,  2.980856E-02,  7.631813E-01],
             [2.447940E-01,  0.000000E+00,  6.141501E-01,  2.368187E-01],
             [5.525472E-01,  4.447026E-03,  3.560413E-01,  0.000000E+00]]
        PI= [0.3, 0.475, 0.225, 0.]

        self.failUnless(allclose(A,  self.weatherhmm.A))
        self.failUnless(allclose(B,  self.weatherhmm.B))
        self.failUnless(allclose(PI, self.weatherhmm.pi))

    def test_04ensemble_average(self):
        print self
        Q_ =[] ; O_ =[]
        for i in self.sampleMultipleSim:
            Q,O = zip(*i)
            Q_.append(Q) ; O_.append(O)

        self.weatherhmm.random_initialisation()
        self.weatherhmm.ensemble_averaging(O_, Q_) # "unit", 1000, 0)
        print self.weatherhmm
        self.weatherhmm.integrity_check()
        A = array([
            [5.145833E-01,  2.979167E-01,  1.041667E-01,  8.333333E-02],
            [3.916667E-01,  3.812500E-01,  1.125000E-01,  1.145833E-01],
            [1.875000E-01,  1.375000E-01,  6.250000E-01,  5.000000E-02],
            [2.500000E-02,  8.750000E-02,  1.000000E-01,  7.875000E-01]
            ])
        B = array([
            [5.301560E-01,  4.419728E-01,  4.945049E-01,  4.928509E-01],
            [8.562561E-02,  2.871941E-01,  2.908651E-01,  2.344170E-01], 
            [3.842184E-01,  2.708331E-01,  2.146300E-01,  2.727321E-01]
            ])
#another observed Bs
# 4.740784E-01  5.717072E-01  3.143790E-01  5.359193E-01 
# 2.746921E-01  2.104418E-01  2.674329E-01  2.017609E-01 
# 2.512295E-01  2.178510E-01  4.181881E-01  2.623198E-01 
#-------------------------------------------------------
# 5.937535E-01  5.297851E-01  3.960208E-01  4.792875E-01 
# 2.827552E-01  2.317921E-01  1.829225E-01  2.349515E-01 
# 1.234913E-01  2.384227E-01  4.210567E-01  2.857610E-01 
        PI = array([0.1, 0.125, 0.525, 0.25])
        self.failUnless( allclose(A,  self.weatherhmm.A)) # leant from states anyway
        #self.failUnless( allclose(B,  self.weatherhmm.B)) # not nearly enough learning for this
        #self.failUnless( allclose(PI, self.weatherhmm.pi)) # same


class TestDeterministic(unittest.TestCase):
    """Viterbi algorithm on a deterministic Markov chain"""
    def setUp(self):
        self.hmm = hmm.HMM(['s1', 's2'], ['o1', 'o2', 'o3'],
                        array([[1.0, 0.0],
                               [0.0, 1.0]]),
                        array([[0.5, 0.0],   # observation of s1
                               [ .5,  .5],   # confusing observation
                               [0.0, 0.5]]), # observation of s2
                        array([0.5, 0.5]))

        self.hmmBias = hmm.HMM(['s1', 's2'], ['o1', 'o2', 'o3'],
                        array([[1.0, 0.0],
                               [0.0, 1.0]]),
                        array([[0.75, 0.00],   # observation of s1
                               [0.25, 0.25],   # confusing, unlikely observation
                               [0.00, 0.75]]), # observation of s2
                        array([0.5, 0.5]))

    def test_viterbi(self):
        observations = ['o1'] * 5
        result = ['s1'] * 5
        self.assertEqual(self.hmm.viterbi(observations), self.hmm.viterbi_log(observations))
        self.assertEqual(self.hmm.viterbi(observations), result)
        print observations, result
        _hmm = self.hmm.transition_likelihood_simple(self.hmm.viterbi(observations), observations)

        self.assertEqual(self.hmmBias.viterbi(observations), self.hmmBias.viterbi_log(observations))
        self.assertEqual(self.hmmBias.viterbi(observations), result)
        _hmmBias = self.hmmBias.transition_likelihood_simple(self.hmmBias.viterbi(observations), observations)
        print _hmm, _hmmBias
        assert (_hmm < _hmmBias)

        observations = ['o1'] * 7
        result = ['s1'] * 7
        self.assertEqual(self.hmm.viterbi(observations), self.hmm.viterbi_log(observations))
        self.assertEqual(self.hmm.viterbi(observations), result)
        print observations, result
        _hmm = self.hmm.transition_likelihood_simple(self.hmm.viterbi(observations), observations)

        self.assertEqual(self.hmmBias.viterbi(observations), self.hmmBias.viterbi_log(observations))
        self.assertEqual(self.hmmBias.viterbi(observations), result)
        _hmmBias = self.hmmBias.transition_likelihood_simple(self.hmmBias.viterbi(observations), observations)
        print _hmm, _hmmBias
        assert (_hmm < _hmmBias)

        observations = ['o3'] * 5
        result = ['s2'] * 5
        self.assertEqual(self.hmm.viterbi(observations), self.hmm.viterbi_log(observations))
        self.assertEqual(self.hmm.viterbi(observations), result)
        print observations, result
        _hmm = self.hmm.transition_likelihood_simple(self.hmm.viterbi(observations), observations)

        self.assertEqual(self.hmmBias.viterbi(observations), self.hmmBias.viterbi_log(observations))
        self.assertEqual(self.hmmBias.viterbi(observations), result)
        _hmmBias = self.hmmBias.transition_likelihood_simple(self.hmmBias.viterbi(observations), observations)
        print _hmm, _hmmBias
        assert (_hmm < _hmmBias)

        observations = ['o2'] + ['o1'] * 4
        result = ['s1'] * 5
        self.assertEqual(self.hmm.viterbi(observations), self.hmm.viterbi_log(observations))
        self.assertEqual(self.hmm.viterbi(observations), result)
        print observations, result
        _hmm = self.hmm.transition_likelihood_simple(self.hmm.viterbi(observations), observations)

        self.assertEqual(self.hmmBias.viterbi(observations), self.hmmBias.viterbi_log(observations))
        self.assertEqual(self.hmmBias.viterbi(observations), result)
        _hmmBias = self.hmmBias.transition_likelihood_simple(self.hmmBias.viterbi(observations), observations)
        print _hmm, _hmmBias
        assert (_hmm < _hmmBias)

        observations = ['o3'] + ['o2'] * 4
        result = ['s2'] * 5
        self.assertEqual(self.hmm.viterbi(observations), self.hmm.viterbi_log(observations))
        self.assertEqual(self.hmm.viterbi(observations), result)
        print observations, result
        _hmm = self.hmm.transition_likelihood_simple(self.hmm.viterbi(observations), observations)

        self.assertEqual(self.hmmBias.viterbi(observations), self.hmmBias.viterbi_log(observations))
        self.assertEqual(self.hmmBias.viterbi(observations), result)
        _hmmBias = self.hmmBias.transition_likelihood_simple(self.hmmBias.viterbi(observations), observations)
        print _hmm, _hmmBias
        assert (_hmm > _hmmBias)

        observations = ['o2', 'o3', 'o2', 'o2', 'o3']
        result = ['s2'] * 5
        self.assertEqual(self.hmm.viterbi(observations), self.hmm.viterbi_log(observations))
        self.assertEqual(self.hmm.viterbi(observations), result)
        print observations, result
        _hmm = self.hmm.transition_likelihood_simple(self.hmm.viterbi(observations), observations)

        self.assertEqual(self.hmmBias.viterbi(observations), self.hmmBias.viterbi_log(observations))
        self.assertEqual(self.hmmBias.viterbi(observations), result)
        _hmmBias = self.hmmBias.transition_likelihood_simple(self.hmmBias.viterbi(observations), observations)
        print _hmm, _hmmBias
        assert (_hmm > _hmmBias)

        observations = ['o2'] * 4 + ['o1']
        result = ['s1'] * 5
        self.assertEqual(self.hmm.viterbi(observations), self.hmm.viterbi_log(observations))
        self.assertEqual(self.hmm.viterbi(observations), result)
        print observations, result
        _hmm = self.hmm.transition_likelihood_simple(self.hmm.viterbi(observations), observations)

        self.assertEqual(self.hmmBias.viterbi(observations), self.hmmBias.viterbi_log(observations))
        self.assertEqual(self.hmmBias.viterbi(observations), result)
        _hmmBias = self.hmmBias.transition_likelihood_simple(self.hmmBias.viterbi(observations), observations)
        print _hmm, _hmmBias
        assert (_hmm > _hmmBias)

        observations = ['o2'] * 4 + ['o3']
        result = ['s2'] * 5
        self.assertEqual(self.hmm.viterbi(observations), self.hmm.viterbi_log(observations))
        self.assertEqual(self.hmm.viterbi(observations), result)
        print observations, result
        _hmm = self.hmm.transition_likelihood_simple(self.hmm.viterbi(observations), observations)

        self.assertEqual(self.hmmBias.viterbi(observations), self.hmmBias.viterbi_log(observations))
        self.assertEqual(self.hmmBias.viterbi(observations), result)
        _hmmBias = self.hmmBias.transition_likelihood_simple(self.hmmBias.viterbi(observations), observations)
        print _hmm, _hmmBias
        assert (_hmm > _hmmBias)

# impossible
        observations = ['o2'] * 4 + ['o3'] + ['o1']
        print self.hmm.viterbi(observations), "(impossible path)"
        print self.hmm.transition_likelihood_simple(self.hmm.viterbi(observations), observations)
        print self.hmmBias.transition_likelihood_simple(self.hmm.viterbi(observations), observations)
        self.assertEqual(
            self.hmm.transition_likelihood_simple(self.hmm.viterbi(observations), observations),
            0)
        self.assertEqual(
            self.hmmBias.transition_likelihood_simple(self.hmm.viterbi(observations), observations),
            0)




class TestBaum_Welch(unittest.TestCase): #pending checks
    """Test the Baum-welch algorithm"""

    def setUp(self):
        self.aHMM = hmm.HMM( ['s1', 's2'], ['o1', 'o2', 'o3'])
        
        self.aHMM_1 = hmm.HMM( ['s1', 's2'], ['o1', 'o2', 'o3'], 
                            array([[0.7, 0.3], [0.2, 0.8]]),
                            array([[0.2, 0.4], [0.6, 0.2], [0.2, 0.4]]),
                            array([0.2, 0.8]))
        self.det = hmm.HMM(['s1'], ['o1', 'o2'])
        self.test = hmm.HMM( range(5), range(5) )
        self.det2 = hmm.HMM(['s1', 's2'], ['o1', 'o2'] )

    def _learn_compare(self, chain):      
        self.aHMM.baum_welch_o(chain)
        
    def test_baumwelch_1(self):
        """test the observations (1,2,1,2,1,2,1,2,1,2) """
        chain = ['o1', 'o2'] * 5 
        self._learn_compare(chain)

    def test_baumwelch_2(self):
        """test the observations (1,1,1,1,1,2,2,2,2,2) """
        chain =  ['o1'] * 5 + ['o2'] * 5
        self._learn_compare(chain)

    def test_baumwelch_3(self):
        """test the observations (3,3,3,3,3,3,3,3,3,1) """
        chain = ['o3'] * 9 + ['o1']
        self._learn_compare(chain)

    def test_baumwelch_4(self):
        """test the observations (1,2,1,2,1,2,1,2,1,2) """
        chain = ['o1', 'o2'] * 5 
        self._learn_compare(chain)

    def test_baumwelch_6(self):
        chain = ['o2'] * 2
        resA = self.det.A
        resB = array([[0.], [1.]])
        respi = self.det.pi
        self.det.baum_welch_o(chain)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

    def test_baumwelch_7(self):
        observation = self.test.simulate(10)
        self.test.random_initialisation()
        self.test.baum_welch_o(observation)
        self.test.integrity_check()

    def test_baum_welch_multiple_1(self):
        chains = []
        for i in xrange(10):
            chains.append(self.aHMM.simulate(10))
        A = self.aHMM.A
        B = self.aHMM.B
        PI = self.aHMM.pi
        self.aHMM.baum_welch_multiple(chains)
        self.aHMM.integrity_check()
        self.failUnless( allclose(self.aHMM.A, A))
        self.failUnless( allclose(self.aHMM.B, B))
        self.failUnless( allclose(self.aHMM.pi, PI))

    def test_baum_welch_multiple_2(self):
        chains = [ ['o1','o2','o2','o2','o2'], ['o1','o2','o2','o2','o2','o2','o2'],
                    ['o2','o2','o2','o2','o2','o2','o2']]
        self.aHMM_1.baum_welch_multiple(chains)
        self.aHMM_1.integrity_check()

    def test_baum_welch_multiple_3(self):
        chains = [ ['o1','o2','o2','o2','o2'], ['o1','o2','o2','o2','o2','o2','o2'],
                    ['o2','o2','o2','o2','o2','o2','o2']]
        self.aHMM_1.baum_welch_multiple(chains)
        self.aHMM_1.integrity_check()

    def test_baum_welch_multiple_4(self):
        chains = [ ['o2'] * 2, ['o2'] * 3, ['o2'] * 4]
        resA = self.det.A
        resB = array([[0.], [1.]])
        respi = self.det.pi
        self.det.baum_welch_multiple(chains)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

    def test_baum_welch_multiple_5(self):
        chains = []
        for i in xrange(10):
            chains.append(self.test.simulate(20))
        self.test.random_initialisation()
        self.test.baum_welch_multiple(chains)
        self.test.integrity_check()


class TestEnsembleAveraging(unittest.TestCase): #pending checks
    def setUp(self):
        self.det = hmm.HMM(['s1'], ['o1', 'o2'])
        self.test = hmm.HMM( ['s1', 's2'], ['o1', 'o2'] )
        self.gen = hmm.HMM( ['s1', 's2'], ['o1', 'o2'],
                            array([[0.7, 0.3], [0.2, 0.8]]),
                            array([[0.2, 0.6], [0.8, 0.4]]),
                            array([0.5, 0.5]))
        self.aHMM = hmm.HMM(['s1', 's2'], ['o1', 'o2'])

    def test_ens_average_1(self):
        set_observations = [ ['o2'] * 2, ['o2'] * 3, ['o2'] * 4]
        resA = self.det.A
        resB = array([[0.], [1.]])
        respi = self.det.pi
        self.det.ensemble_averaging(set_observations, "unit", 1000, 0)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

    def test_ens_average_2(self):
        chains = []
        for i in xrange(10):
            chains.append(self.gen.simulate(10))
        self.test.ensemble_averaging(chains, "unit", 1000, 1)
        self.test.integrity_check()

    def test_ens_average_3(self):
        chains = [ ['o1', 'o2', 'o2', 'o2', 'o2'], ['o1', 'o2', 'o2', 'o2', 'o2'],
                        ['o2','o2','o2','o2','o2','o2','o2']]
        self.aHMM.ensemble_averaging(chains, "unit", 1000, 0)
        self.aHMM.integrity_check()

    def test_ens_average_4(self):
        set_observations = [ ['o2'] * 2, ['o2'] * 3, ['o2'] * 4]
        resA = self.det.A
        resB = array([[0.], [1.]])
        respi = self.det.pi
        self.det.ensemble_averaging(set_observations, "Pall", 1000, 0)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

    def test_ens_average_5(self):
        set_observations = [ ['o2'] * 2, ['o2'] * 3, ['o2'] * 4]
        resA = self.det.A
        resB = array([[0.], [1.]])
        respi = self.det.pi
        self.det.ensemble_averaging(set_observations, "Pk", 1000, 0)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

class TestWeightingFactor(unittest.TestCase):
    def setUp(self):
        self.hmm1 = hmm.HMM(['s1', 's2'], ['o1', 'o2', 'o3'],
                        array([[0., 1.],
                               [1., 0.]]),
                        array([[0.5, 0.0],
                               [ .5,  .5],
                               [0.0, 0.5]]),
                        array([0.5, 0.5]))
        self.hmm2 = hmm.HMM(['s1', 's2'], ['o1', 'o2'],
                        array([[0., 1.],
                               [0., 1.]]),
                        array( [[1., 0.],
                                [0., 1.]] ),
                        array([0.5, 0.5]))

    def test_Weighting_factor_Pall_1(self):
        set_obs = [['o1', 'o2'], ['o2', 'o2']]
        resP = 1./4
        P = self.hmm2._weighting_factor_Pall(set_obs)
        self.failUnless(P == resP)

    def test_Weighting_factor_Pall_2(self):
        set_obs = [['o1', 'o3'], ['o1', 'o2'], ['o2', 'o2']]
        resP = 1./256
        P = self.hmm1._weighting_factor_Pall(set_obs)
        self.failUnless(P == resP)

    def test_Weighting_factor_Pk_1(self):
        obs = ['o1', 'o2']
        resP = 1./2
        P = self.hmm2._weighting_factor_Pk(obs)
        self.failUnless(P == resP)

    def test_Weighting_factor_Pk_2(self):
        obs = ['o1', 'o3']
        resP = 1./8
        P = self.hmm1._weighting_factor_Pk(obs)
        self.failUnless(P == resP)


class TestPickle(unittest.TestCase):  #pending checks, reconsider
    """ test the pickle implementation """
    def setUp(self):
        self.hmm1 = hmm.HMM(['s1', 's2'], ['o1', 'o2', 'o3'])
        self.hmm2 = hmm.HMM(['s1', 's2'], ['o1', 'o2', 'o3'])
        self.hmm2.random_initialisation()
        self.hmm_different = hmm.HMM(['s1'], ['o1'])

    def _compare(self, h1, h2, SaveState=None):
        self.failUnless( allclose(h1.A, h2.A))
        self.failUnless( allclose(h1.B, h2.B))
        self.failUnless( allclose(h1.pi, h2.pi))
        self.failUnless( allclose(h1.N, h2.N))
        self.failUnless( allclose(h1.M, h2.M))
        if SaveState:
            self.failUnless( h1.state_index == h2.state_index)  
            self.failUnless( h1.STATES == h2.STATES)  
            self.failUnless( h1.OBSERVATIONS == h2.OBSERVATIONS)  
            self.failUnless( h1.observation_index == h2.observation_index)      

    def test_pickle(self):
        f = open('./temp/saved_hmm', 'w')
        self.hmm1.saveHMM(f)
        f.close()
        f = open('./temp/saved_hmm', 'r')
        self.hmm2.loadHMM(f)
        f.close()
        self.hmm2.integrity_check()
        self._compare(self.hmm1, self.hmm2)
        os.remove('./temp/saved_hmm')

        f = open('./temp/saved_hmm', 'w')
        self.hmm1.saveHMM(f, 1)
        f.close()
        f = open('./temp/saved_hmm', 'r')
        self.hmm_different.loadHMM(f)
        f.close()
        self.hmm_different.integrity_check()
        self._compare(self.hmm1, self.hmm_different)
        os.remove('./temp/saved_hmm')

        # Test saving and loading
        test = hmm.HMM(['s'+ str(i) for i in range(1,4)], 
                       ['o'+ str(i) for i in range(1,5)])
        test.random_initialisation()
        f = open('./temp/saved_hmm', 'w')
        test.saveHMM(f, 1)
        f.close()
        f = open('./temp/saved_hmm')
        toload = hmm.HMM(['s'], ['o'])
        toload.loadHMM(f)
        print "test io"
        print toload

##suite = unittest.TestLoader().loadTestsFromTestCase(TestFunctions)

suite = unittest.TestLoader().loadTestsFromNames([
        "test_hmm.TestBasicOperation_crazyCoin",
        "test_hmm.TestBasicOperation_fairCoin",
        "test_hmm.TestFunctions",
        "test_hmm.TestStates",
        "test_hmm.TestDeterministic",
        "test_hmm.TestBaum_Welch",
        "test_hmm.TestWeightingFactor",
        "test_hmm.TestEnsembleAveraging",
        "test_hmm.TestPickle",
        ])

if __name__ == '__main__':
    #unittest.main(verbosity=2)   # python 2.7 required
    #unittest.main()
    unittest.TextTestRunner(verbosity=3).run(suite)

