# Objective
To understand how raw texts are converted to tokens and embeddings and how contextual representation increases similarity between two different sentences as compared to static representation.

---

# Models and Tokenizers Used
1. ``GPT2``, ``GPT2Tokenizer``
2. ``Qwen/Qwen2.5-0.5B`` - Both tokenizer and model
3. ``bert-base-uncased`` - Both tokenizer and model

---

# Experiments
1. Import GPT2 tokenizer from ``huggingface`` and check the vocabulary size, maximum context window, model inputs, dimensions and special tokens used.
2. Use GPT2 tokenizer to tokenize a single word and convert the tokens to token ids.
3. Use GPT2 tokenizer to convert a single sentence, tokenize the entire sequence, convert the token into token ids.
4. Testing sub-word tokenization - testing if the GPT2 splits different words by stems and suffixes.
5. Tokenize rarer words using GPT2 to check if ``eos_token`` and ``unk_token`` are working.
6. Import ``Qwen/Qwen2.5-0.5B`` tokenizer from ``huggingface`` and  check the vocabulary size, maximum context window, model inputs, dimensions and special tokens used. Also, test how this tokenizer works.
7. Compare ``Qwen/Qwen2.5-0.5B`` tokenizer and ``GPT2Tokenizer`` to check how they both differ in tokens and token ids for the same sentence.
8. Import ``bert-base-uncased`` tokenizer from huggingface to check the tokenizer details and compare it with ``GPT2Tokenizer`` and ``Qwen/Qwen2.5-0.5B`` to see how it differs from them.
9. Import the three models from ``huggingface`` and compute the embeddings matrix, input ids using token ids and check the shapes, dtype and runtime device of each embedding matrix.
10. Use all the three models to compute similarities between five different words using cosine_similarity on embeddings and create a matrix to display the cosine similarities between each pair and see how each similarity matrix differs from the other.
11. Use all the three models to compute similarities between five different sentences using cosine_similarity on embeddings and create a matrix to display the cosine similarities between each pair of sentences and notice the difference in similarity scores as compared to single word similarity.

---

# Observations
## Experiments 1-3 (Tokenizer Inspection)
- Out of the three tokenizers, ``Qwen/Qwen2.5-0.5B`` has the largest vocabulary size of 151643.
- `GPT2Tokenizer` has three special tokens, ``Qwen/Qwen2.5-0.5B`` has 2 special tokens and ``bert-base-uncased`` has 5 special tokens. GPT2 and Qwen has only one similar special token, i.e., `eos_token = <|endoftext|>` and the rest of the tokens are different.
- Each tokenizer has different maximum context lengths with bert and gpt2 having the lowest and qwen having the highest.

## Experiments 4-5 (Subword Tokenization)
This box shows the sub-word tokenization of 5 similar words.
```
words = ['program', 'programmer', 'programming', 'programmed', 'programs']
 
words -> token -> token_id <- No. of tokens
 
program -> ['program'] -> [23065]     <- 1 tokens
programmer -> ['program', 'mer'] -> [23065, 647]     <- 2 tokens
programming -> ['program', 'ming'] -> [23065, 2229]     <- 2 tokens
programmed -> ['program', 'med'] -> [23065, 1150]     <- 2 tokens
programs -> ['program', 's'] -> [23065, 82]     <- 2 tokens
```
- The `GPT2 Tokenizer` considered program as a stem and the suffixes were seperated from it.
- The word `program` stayed as one token, whereas other words which had extended suffixes were converted to two tokens.
- Also, the tokenizer produced consistent token id for the word program.
  
Similarly, this box shows the sub-word tokenization of rarer words.
```
rare_words = ["electroencephalography", "antidisestablishmentarianism", "electromagnetohydrodynamics"]
 
electroencephalography -> ['elect', 'ro', 'ence', 'phal', 'ography'] <- 5 tokens
antidisestablishmentarianism -> ['ant', 'idis', 'establishment', 'arian', 'ism'] <- 5 tokens
electromagnetohydrodynamics -> ['elect', 'rom', 'ag', 'net', 'ohyd', 'rod', 'ynam', 'ics'] <- 8 tokens
```
- The `GPT2 Tokenizer` split the rarer words into many small tokens.
- The three words break into 5, 5, and 8 tokens respectively.
- This shows that in sub-word tokenization, longer or more complex words are split into more subword pieces, especially for uncommon terms like electromagnetohydrodynamics.

Finally, from these tests the unk_token, eos_token of `GPT2 Tokenizer` did not appear, suggesting that sub-word tokenization is effective for OOV problem.

## Experiments 6-8 (Tokenizer Comparison)
- `bert-base-uncased` lower-cased the tokens whereas `GPT2 Tokenizer` and `Qwen/Qwen2.5-0.5B` kept the tokens casing same as the words.
- Even though the tokens looked similar, the token ids were different.
- Across 106 long and rare sentences, the three tokenizers produced broadly similar token counts, with only minor variations. GPT-2 and Qwen tokenizers generally generated comparable sequence lengths, while BERT sometimes produced slightly fewer or slightly more tokens depending on its WordPiece segmentation. Overall, no tokenizer consistently produced the shortest or longest tokenized sequences across all inputs.

## Experiments 9-11 (Embeddings and similarities)
- `GPT2 Tokenizer` and `bert-base-uncased` have the same hidden size, hence the last dimension stays the same. What changes is the sequence length, because each tokenizer may split the sentence differently and may add different special tokens.
- `Qwen/Qwen2.5-0.5B` has different embedding shape from these two tokenizers.

Here is a similarity matrix made from comparing the similarities between embeddings produced by GPT2 using 5 words:
```
The 5 words: "cat", "dog", "banana", "apple", "rainbow"

Similarity Matrix:
            cat       dog    banana     apple   rainbow
cat      1.000000  0.381537  0.267502  0.247951  0.276125
dog      0.381537  1.000000  0.237725  0.274597  0.348723
banana   0.267502  0.237725  1.000000  0.286307  0.409835
apple    0.247951  0.274597  0.286307  1.000000  0.326594
rainbow  0.276125  0.348723  0.409835  0.326594  1.000000
```

Three separate similarity matrices were generated by comparing sentence embeddings within each model, with no correlation between the models themselves.
```
The 5 sentences:
s1 -> The cat sat on the mat.
s2 -> A cat was sitting on the rug.
s3 -> She finished her homework quickly.
s4 -> She completed her assignment fast.
s5 -> The sun is shining brightly.

Similarity Matrix(GPT2):
    s1        s2        s3        s4        s5
s1  1.000000  0.906041  0.691473  0.685445  0.756417
s2  0.906041  1.000000  0.721407  0.707248  0.777304
s3  0.691473  0.721407  1.000000  0.884238  0.694014
s4  0.685445  0.707248  0.884238  1.000000  0.675286
s5  0.756417  0.777304  0.694014  0.675286  1.000000

Similarity Matrix(Qwen2.5-0.5B):
    s1        s2        s3        s4        s5
s1  0.996094  0.746094  0.190430  0.214844  0.404297
s2  0.746094  1.000000  0.208008  0.211914  0.433594
s3  0.190430  0.208008  1.000000  0.796875  0.227539
s4  0.214844  0.211914  0.796875  1.000000  0.208984
s5  0.404297  0.433594  0.227539  0.208984  1.000000

Similarity Matrix(bert-base-uncased):
    s1        s2        s3        s4        s5
s1  1.000000  0.904657  0.670176  0.685214  0.703232
s2  0.904657  1.000000  0.705342  0.724867  0.752007
s3  0.670176  0.705342  1.000000  0.862485  0.708547
s4  0.685214  0.724867  0.862485  1.000000  0.700706
s5  0.703232  0.752007  0.708547  0.700706  1.000000
```

- The similarity matrix built from the 5 sentences showed clearer and more meaningful patterns than the word matrix.
- The similarities found in the word matrix are inaccurate and shows how the static embedings fail to capture meanings.
- The sentence-level similarities were much higher as compared to the word similarities. This suggests that contextual representation capture meanings better than static embeddings.
- `Qwen-2.5-0.5B` shows clearer clustering, with high similarity inside related sentence pairs and much lower similarity across unrelated ones. `GPT-2` and bert gives generally high similarity values for almost all sentences, so its pattern is smoother and less separated. This means Qwen seems more selective in capturing meaning, while `GPT-2` makes many sentences look broadly similar.
- Although `GPT-2` and `bert-base-uncased` have the same hidden size of 768, their similarity patterns are still shaped by model behavior rather than vector size alone. `Qwen2.5-0.5B`, with hidden size 896, produces a more sharply clustered matrix, suggesting a more separated embedding space for these sentences.

# Key Learnings
- Tokenizers are important as they convert raw texts into tokens and token ids before sending them to the embedding layer.
- All models have different vocabulary size and id patterns due to the way the models were trained, therefore, all models do not tokenize the same sentence identically.
- Sub-words breaks down words into chunks of texts, which helps in avoiding the `Out-Of-Vocabulary` problem on encountering an unknown word.
- Vocabulary size matters, but it does not decide model quality by itself. Context handling, tokenizer design, hidden size, and architecture also shape the embeddings and similarity patterns.
- Contextual embeddings captures the relationships between each word with the every other words in the sequence, whereas static embeddings are just representation of a single token itself.
- As sentence embeddings learns contexts of each word all the other words in the sequence, the embeddings captures a much better pattern and meaning of the sentences.
- Two sentences with different words can become similar because sentences embeddings cpatures the full sentence context and if the contextual embeddings between the two sentences are similar, then, they are considered as similar.

# Engineering Takeaways
- Different tokenizers have different vocabularies sizes and ids. These different token ids affects the embedding matrix's results. A same model can generate different embeddings based on the different tokenizers used.
- However, mixing tokenizers across models is not recommended, since it might change how the models interpret context. If a different tokenizer is used on a model, the embedding layer receives wrong token ids and the model produces inaccurate contextual embeddings resulting in broken embeddings.
- Tokens are the unit of text processed by the LLMs, not raw words. The LLMs receives input tokens and produces output tokens.LLM APIs costs are token dependent. Hence, tokens count matter for LLM applications.
- Embeddings are useful beyond language modelling because the same idea works across domains like maths, multimodal AI, etc. Embeddings converts texts, images or other data into vectors which capture meaning and relationships  which is much more flexible than keyword matching alone.
- Cosine similarity can be used to compute the similarities between embeddings of different kinds of data which helps in understanding the realtionships between them.
- NLP helps computers understand, analyze, and process human language, while contextual embeddings provide meaningful representations that models can use for tasks such as classification, translation, and summarization.
- Embedding dimension affects how much information a vector can store and can influence how well a model separates meanings, but the model’s behavior also depends heavily on training data, architecture, and tokenizer design.
- A larger context window allows the models to accept longer inputs and produce a richer contextual representation from them.