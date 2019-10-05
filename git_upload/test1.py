dictionary = { 1: 'one', 2:'two', 3:'three' }
print(dictionary)
dictionary['ONE'] = dictionary.pop(1)
dictionary[1] = dictionary.pop('ONE')
print(sorted(dictionary))
print(dictionary)