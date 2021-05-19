# import io
# e = model.layers[0]
# weights = e.get_weights()[0]
# weights.shape
# out_v = io.open("vector_data.tsv","w",encoding="utf-8")
# out_m = io.open("meta_data.tsv","w",encoding="utf-8")
#
# for i in range(1,30000):
#     word = tokenize.index_word[i]
#     vec_tor = weights[i]
#     out_m.write(word +"\n")
#     out_v.write('\t'.join(str(x) for x in vec_tor) +"\n")
