from sentence_transformers import SentenceTransformer, util
import torch

device = "cuda:2"
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

col1 = ["我爱你", "我爱你", "你是傻瓜", "你好"]
col2 = ["I love you", "I like you", "I love apple", "thank you"]

vectors1 = model.encode(col1, convert_to_tensor=True, device=device)
vectors2 = model.encode(col2, convert_to_tensor=True, device=device)


print(vectors1)
print(vectors2)
print(vectors1.requires_grad)
print(vectors1.shape)

cosine_scores = util.cos_sim(vectors1, vectors2)

for i, (sent1, sent2) in enumerate(zip(col1, col2)):
    if cosine_scores[0][i] >= 0.5:
        label = "Similar"
    else:
        label = "Not Similar"
    print("sentence1: {} | sentence2: {} | score: {} | prediction: {}".format(sent1, sent2, cosine_scores[0][i], label))
