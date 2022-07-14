import torch
from transformers import T5Model, T5Tokenizer
from numba import njit
from torch import nn


model = T5Model.from_pretrained("t5-large")
tok = T5Tokenizer.from_pretrained("t5-large")

print("Finish loading")


@torch.no_grad()
@njit()
def t5_sentence_embedding(model, tok, sentence):
    enc = tok(sentence, return_tensors="pt")
    output = model.encoder(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        return_dict=True
    )
    emb = output.last_hidden_state
    sen_embedding = torch.mean(emb, dim=1)
    return sen_embedding


def summarize(model, tok, text):
    preprocess_text = text.strip().replace('\n', '')
    t5_prepared_text = 'summarize: ' + preprocess_text
    tokenized_text = tok.encode(t5_prepared_text, return_tensors="pt")
    summary_ids = model.generate(input_ids=tokenized_text["input_ids"],
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=1,
                                 max_length=100,
                                 early_stopping=False)
    output = tok.decode(summary_ids[0], skip_special_tokens=True)
    return output


sen1 = ['I love you']
sen2 = ['我爱你']
sen3 = ['You are a dog']
sen4 = ['happy new year']
sen5 = ['you love I']
cos = torch.nn.CosineSimilarity(dim=1)

print(cos(t5_sentence_embedding(model, tok, sen1), t5_sentence_embedding(model, tok, sen2)))
print(cos(t5_sentence_embedding(model, tok, sen1), t5_sentence_embedding(model, tok, sen3)))
print(cos(t5_sentence_embedding(model, tok, sen1), t5_sentence_embedding(model, tok, sen4)))
print(cos(t5_sentence_embedding(model, tok, sen2), t5_sentence_embedding(model, tok, sen3)))
print(cos(t5_sentence_embedding(model, tok, sen2), t5_sentence_embedding(model, tok, sen4)))
print(cos(t5_sentence_embedding(model, tok, sen3), t5_sentence_embedding(model, tok, sen4)))
print(cos(t5_sentence_embedding(model, tok, sen1), t5_sentence_embedding(model, tok, sen5)))
