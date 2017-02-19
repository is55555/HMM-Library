# -*- coding: utf-8 -*-
import hmm

from numpy import array

HMM = hmm.HMM

def test1_viterbi( h, chain ):
    """simple test"""
    print "Chain      : ", chain
    print "analyse    : ", h.viterbi(chain)
    print "analyse_log: ", h.viterbi_log(chain)


def test1():
    """Simple test, that will check if the viterbi algorithm
    correctly determine the hidden states of a HMM."""
    test = HMM(['a', 'b'], ['s1', 's2', 's3'],
               array([[.3, .7], [.5, .5]]),
               array([[.5, 0], [.5, .5], [0, .5]]),
               array([.9, .1]))
    test.dump()
    test1_viterbi(test, ['s1'] * 3)
    test1_viterbi(test, ['s1'] * 3 + ['s3'] * 3)
    test1_viterbi(test, ['s1', 's2', 's3'] * 3)

    test.random_initialisation()
    test.dump()
    test1_viterbi(test, ['s1'] * 3)
    test1_viterbi(test, ['s1'] * 3 + ['s3'] * 3)
    test1_viterbi(test, ['s1', 's2', 's3'] * 3)


def test2():
    """This test will display the computed likelyhood of some sentences given
    some predetermined transition and observation matrices"""
    nom = 'soleil ville parc chat chien jaune souris poule jardin bec griffe sel poulet poivre'.split()
    verbe = 'brillait mange chasse court dort griffe est ressemble'.split()
    adj = 'grand petit gentil endormi jaune grande petite gentille endormie'.split()
    adv = 'vigoureusement rapidement vite'.split()
    det = 'le la les un une des'.split()
    pro = 'je tu il elle on nous vous ils elles le la les lui moi toi eux'.split()
    pre = 'à pour sur sous près de du au avec sans'.split()
    univers = []
    for mot in nom + verbe + adj + adv + det + pro + pre:
        univers.append(mot)
    test = HMM(['adj', 'nom', 'verbe', 'adv', 'det', 'pro', 'pre'], univers)
    test.A[:,:] = 0.0 # clear transition proba
    test.set_A('det', 'adj', .5)
    test.set_A('det', 'nom', .5)
    test.set_A('nom', 'adj', .2)
    test.set_A('nom', 'verbe', .2)
    test.set_A('nom', 'nom', .2)
    test.set_A('nom', 'pro', .2)
    test.set_A('nom', 'adv', .1)
    test.set_A('nom', 'pre', .1)
    test.set_A('pro', 'adj', .2)
    test.set_A('pro', 'verbe', .2)
    test.set_A('pro', 'nom', .2)
    test.set_A('pro', 'pro', .2)
    test.set_A('pro', 'adv', .1)
    test.set_A('pro', 'pre', .1)
    test.set_A('adj', 'adj', .2)
    test.set_A('adj', 'nom', .6)
    test.set_A('adj', 'pre', .1)
    test.set_A('adj', 'verbe', .1)
    test.set_A('pre', 'det', .8)
    test.set_A('pre', 'nom', .2)
    test.set_A('verbe', 'verbe', .2)
    test.set_A('verbe', 'adv', .2)
    test.set_A('verbe', 'det', .3)
    test.set_A('verbe', 'pre', .3)
    test.set_A('adv', 'pre', .3)
    test.set_A('adv', 'verbe', .4)
    test.set_A('adv', 'det', .3)
    test.integrity_check()
    for liste, state in [ (nom, 'nom'), (verbe, 'verbe'), (adj, 'adj'),
                          (adv, 'adv'), (det, 'det'), (pro, 'pro'),
                          (pre, 'pre')]:
        taille = len(liste)
        proba = 1.0 / taille
        for mot in liste:
            test.set_B(state, mot, proba)
        test.set_PI(state, 1. / 7)

    phrases = ('un grand soleil jaune brillait vigoureusement sur la ville endormie',
               'le petit chat mange des souris',
               'je mange du poulet au poivre sans sel',
               )

    for p in phrases:
        p = p.split()
        a = test.viterbi(p)
        for i in range(len(p)):
            p[i] = (p[i], a[i])
        print p

if __name__ == "__main__":
        print "test1"
        test1()
        print "---------------------------"
        print "test2"
        test2()
