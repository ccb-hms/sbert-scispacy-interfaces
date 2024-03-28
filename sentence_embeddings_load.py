import pickle

with open("output_sbert/efo_embeddings.pkl", "rb") as input_pickle:
    pickled_data = pickle.load(input_pickle)
    corpus_sentences = pickled_data['sentences']
    corpus_embeddings = pickled_data['embeddings']

    print(corpus_sentences[0])
    print(corpus_embeddings[0])

    print()
    print(corpus_sentences[201])
    print(corpus_embeddings[201])
