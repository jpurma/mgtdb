
import mgtdbp as original
import mgtdbpA as changed
import mg0 as grammar

sentences = ["the king prefers the beer", "which king says which queen knows which king says which wine the queen prefers"]


for sentence in sentences:
    results1 = original.go1(grammar.g, 'C', -0.0001, sentence=sentence)
    gr = changed.Grammar(grammar.g, 'C', -0.0001, sentence=sentence)
    results2 = gr.results
    for key in sorted(list(results1.keys())):
        if results1[key] == results2[key]:
            print ('OK (%s)' % key)
        else:
            print('*** fail (%s): ' % key)
            print(results1[key])
            print('*** vs. ***')
            print(results2[key])

