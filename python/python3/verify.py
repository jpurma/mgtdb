
import mgtdbp as original
import mgtdbpA as changed
import mg0 as grammar

sentences = ["the king prefers the beer"]

for sentence in sentences:
    results1 = original.go1(grammar.g, 'C', -0.0001, sentence=sentence)
    results2 = changed.go1(grammar.g, 'C', -0.0001, sentence=sentence)
    for key in sorted(list(results1.keys())):
        if results1[key] == results2[key]:
            print ('OK (%s)' % key)
        else:
            print('*** fail: ')
            print(results1[key])
            print('*** vs. ***')
            print(results2[key])

