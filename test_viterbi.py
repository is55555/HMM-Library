# -*- coding: utf-8 -*-

# I think I took something from a book or from a website where they went through examples, but I cannot remember

import hmm

from numpy import array

HMM = hmm.HMM

def test1_viterbi( h, chain ):
    """simple test"""
    print "Chain      : ", chain
    print "analysis    : ", h.viterbi(chain)
    print "analysis_log: ", h.viterbi_log(chain)


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
    """Display the computed likelihood of some sentences in Spanish
    given some predetermined transition and observation matrices.
    Note that without a proper stemmer this is mostly just a toy.
    Let alone without a proper vocabulary.

    Also, more detailed parts of speech would be a lot more accurate. For instance for the different kinds of pronoun."""

    phrases = ('el rápido zorro marrón salta sobre el perro perezoso',
               'este buitre voraz de ceño torvo',
               'me devora las entrañas fiero',
               'es mi único constante compañero',)


    suj = 'zorro perro buitre ceño entrañas compañero Roque rabo'.split() # sujeto/sustantivo/nombre (noun) just for sample usage
    ver = 'salta devora es tiene'.split() # same, just for the sample sentences
    adj = 'voraz torvo perezoso marrón rápido fiero único'.split() #
    adv = 'no constante lentamente rápidamente veloz raudo mucho poco bastante bien mal arriba abajo ahí allí'.split() # some samples
    det = 'el la los las un una unos'.split()
    pro = 'yo tú él ella usted vos nosotros nosotras vosotros vosotras ustedes ellos ellas' # personales tónicos
    pro+= 'consigo me te se lo la le se nos os se los las' # personales
    pro+= 'mío mía míos mías tuyo tuya tuyos tuyas suyo suya suyos suyas nuestro nuestra nuestros nuestras vuestro vuestra vuestros vuestras' # posesivos
    pro+= 'este	esta esto estos estas ese esa eso esos esas aquel aquella aquello aquellos aquellas' # demostrativos
    pro = pro.split()
    pre = 'a ante bajo cabe con contra de desde en entre hacia hasta para por según sin so sobre tras'.split()
    universo = []
    for palabra in suj + ver + adj + adv + det + pro + pre:
        universo.append(palabra)
    test = HMM(['adj', 'suj', 'ver', 'adv', 'det', 'pro', 'pre'], universo)
    test.A[:,:] = 0.0 # clear transition probabilities
    test.set_A('det', 'adj', .5) # hand made
    test.set_A('det', 'suj', .5)
    test.set_A('suj', 'adj', .2)
    test.set_A('suj', 'ver', .2)
    test.set_A('suj', 'suj', .2)
    test.set_A('suj', 'pro', .2)
    test.set_A('suj', 'adv', .1)
    test.set_A('suj', 'pre', .1)
    test.set_A('pro', 'adj', .2)
    test.set_A('pro', 'ver', .2)
    test.set_A('pro', 'suj', .2)
    test.set_A('pro', 'pro', .2)
    test.set_A('pro', 'adv', .1)
    test.set_A('pro', 'pre', .1)
    test.set_A('adj', 'adj', .2)
    test.set_A('adj', 'suj', .6)
    test.set_A('adj', 'pre', .1)
    test.set_A('adj', 'ver', .1)
    test.set_A('pre', 'det', .8)
    test.set_A('pre', 'suj', .2)
    test.set_A('ver', 'ver', .2)
    test.set_A('ver', 'adv', .2)
    test.set_A('ver', 'det', .3)
    test.set_A('ver', 'pre', .3)
    test.set_A('adv', 'adv', .2)
    test.set_A('adv', 'pre', .2)
    test.set_A('adv', 'ver', .4)
    test.set_A('adv', 'det', .3)
    test.integrity_check()
    for list_, state in [(suj, 'suj'),(ver, 'ver'),(adj, 'adj'),(adv, 'adv'),(det, 'det'),(pro, 'pro'),(pre, 'pre')]:
        probability = 1.0 / len(list_)
        for palabra in list_:
            test.set_B(state, palabra, probability)
        test.set_PI(state, 1. / 7)

    phrases = ('el rápido zorro marrón salta sobre el perro perezoso',
               'este buitre voraz de ceño torvo',
               'me devora las entrañas fiero',
               'es mi único constante compañero',
               'labra mis penas con su pico corvo',
               'el perro de Roque no tiene rabo',
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
