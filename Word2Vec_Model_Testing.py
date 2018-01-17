from gensim.models import KeyedVectors
model = KeyedVectors.load("300features_40minwords_10_2_context")
print(model)
print(model.most_similar("service"))
print(model.most_similar("comcast"))
print(model.most_similar("ajit"))
print(model.doesnt_match("save".split()))
print(model.wv.syn0.shape)
