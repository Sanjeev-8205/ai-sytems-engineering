# Observations
## Logits and Softmax
- The softmax function returned proabilities between 0 and 1 on passign an array as a parameter.
- The tokens having higher logit value has higher probabilities and the tokens having lower logit score has very low probabiity.
- As the softmax fucntion uses exponential to compute probabilities, the difference between the probabilities of logits scores were very big. The higher logits score dominated the probability while the rest had a very small probability.
- Certain tokens were chosen for testing this. On increasing the logit score of those tokens, the probability became high and on decreasing the logit score the probability became low.
```
Original Data:
   Tokens  Logits  Probability
0   Paris     8.2       0.9366
1  London     5.4       0.0570
2  Berlin     2.8       0.0042
3    Rome     2.1       0.0021
4  Banana    -1.4       0.0001

After increasing the original logit score of london by 3.2:
   Tokens  Logits  Probability
0   Paris     8.2     0.400216
1  London     8.6     0.597052
2  Berlin     2.8     0.001808
3    Rome     2.1     0.000898
4  Banana    -1.4     0.000027

Now decreasing the logit score of Paris by 1.5:
   Tokens  Logits  Probability
0   Paris     6.7     0.129593
1  London     8.6     0.866442
2  Berlin     2.8     0.002623
3    Rome     2.1     0.001303
4  Banana    -1.4     0.000039 
```
From this data, it is evident that slight changes in logit scores can cause major changes in the probabilities.

## Greedy Decoding
- The argmax function returns the index of the maximum value of the array. Example: Using the probability from the original data above, the argmax function return 0, since the index 0 has the highest pribability of 93.66%.
- The greedy decoding only selected the token with the highest probability.
- The different temperature settings did not have any effect on the token selection, since, the top probability would still have the highest probability even after applying temperature.

## Temperature
- Used different temperature settings for computing the probabilities. 
- Lower temperature values resulted in high value for the top probabilities while rest of the probabilities were very small. Due to this there were very less noice and more meaningful data. The probabilitiy distribution was steeper and this makes the model more accurate and less affected by randomness.
- Higher temperature settings made the probability distribution flatter, i.e., probability of all tokens became closer to the maximum probability than before and the chances of being selected as an output increases. This resulted in increased randomness. As the choices grew bigger with higher temperature, the chances of the model drifting away from the original response increased as well.
- A lower temperature make the output more deterministic while higher temperature made the model more creative due to diverse choices.
- A very high temperature produced incoherent texts.
- `fig_1_Rank_vs_Probability.jpeg` - This image shows how probability distribution becomes flatter with increasing temperature, which provides good opportunity for other tokens to be selected as well.

## Top-K
- Top-K picks the top `k` probabilities and randomly chooses any one of them.
- It is affected by temperature settings unlike the greedy decoding, if and only if the k>1. If k=1, the model will choose only the output with max probability, which is the same as greedy, and it will remain unaffected by temperature.
- A higher `k` behaves similarly to unrestricted sampling, whereas a lower `k` makes the output restricted.
- If the goal is to generate diverse responses for the same prompt, then, Top-K is more preferrable than greedy.

## Top-P
- The `p` in Top-p refers to probability threshold.
- The probabilities are sorted in descending order and cumulative sum is computed upto `p`.
- The first probability crossing the p is still considered.
- Compared to greedy, top-p is affected by temperature.
- The candidate set size depends on the probability distribution. If a few probabilities are very high, then only a few candidates will be selected. If the proability distribution is more spread out, then, candidate set size increases.
- More adaptive than top-k, produces coherent yet diverse outputs.

## AutoRegressive Generation Using Fake Transformer
The AutoRegressive generation tested the four different techniques: `Greedy`, `Temperature`, `Top-k` and `Top-p` on three different categories. The test used `fake_transformer_1000.json` file to simulate a fake transformer.
- The test was done using 10 different starting words for each loop.
- More than 1000 unique sentences are possible from this json file.

### Category 1 - Unique Sentences

```
k=3, p=0.95, t=0.6

Results:
Greedy = 10 unique sentences
Temperature = 44 unique sentences
Top-k = 53 unique sentences
Top-p = 50 unique sentences

Note: With the same setings, results may vary slightly on each run.
```
  
- Greedy decoding selected 1 different sequence for each starting words, forming a total of 10 unique sentences. This shows that greedy choose only the highest probability sequence only and is very deterministic.
- A moderate temperature of 0.6 was set, and 44 unique sentences were produced. This allowed for unrestricted sampled to some extent and gave a balance between deterministic and non-deterministic.
- Top-k decoding produced a total of 53 unique sentences, which is the most diversed result.
- Top-p decoding produced a total of 50 unique sentences, even with a high threshold of 0.95, which suggest that a few tokens had very high probability and they dominated the output.

### Category 2 - Most Frequent Sentences
```
Greedy - This quick cat found book easily
Temperature - That quick cat found tree today
Top-K - Some quick cat found painting easily
Top-P - That quick cat found tree today
```
- All the techniques produced quite similar most frequent sentences.
- This suggests that the fake transformer is more biased towards few tokens during output generation, i.e., few tokens have higher initial logit scores as compared to other tokens.Therefore, even stochastic decoding methods converge toward a small set of highly probable sentences.

### Category 3 - Shannon Entropy
```
Greedy = 3.3146
Temperature = 4.5912
Top-k = 4.9108
Top-p = 4.8958
```
- Greedy decoding has the lowest shannon entropy, which makes it the most deterministic and is shown by the unique sentences it produced.
- Temperature has a higher entropy than greedy. The sequence generated by temperature decoding are uncertain and diverse and outputs are spread across many sentences.
- Top-k has the highest shannon entropy in this test, with the most diverse sequences and follows a high uncertainty and less deterministic path.
- Top-p has approximately the same entropy as that of top-k, sharing similar characteristics with it in this test.

## Real Inference
Here, a real inference pipeline function was created with all the different decoding techniques using the `Qwen/Qwen2.5-0.5B` model.

### Test - 1
In this test, the inference pipeline was executed on both CPU and GPU, with two different targets - output length and prompt length seperately.
The pipeline was executed only using the greedy decoding technique for this test.
#### Variable Output Lengths
```
Prompt - Explain about Harry Potter and the Philosopher's Stone.
output_lengths = [20, 100, 300, 500]

Results (CPU): Prefill Phase - 2.20%, Decoding Phase - 97.80%
   Output Tokens  total_time_taken      ttfs  prompt_tokens  tokens/s
0             20          4.429964  1.437779           12.0  4.514709
1            100         14.585781  0.592511           12.0  6.855992
2            300         53.642983  0.544685           12.0  5.592530
3            500         66.647152  0.487452           12.0  7.502196

Results (GPU): Prefill Phase - 3.21%, Decoding Phase - 96.79%
   Output Tokens  total_time_taken      ttfs  prompt_tokens   tokens/s
0             20          2.055528  1.084478           12.0   9.729862
1            100          3.700947  0.036644           12.0  27.020109
2            300         11.922450  0.035467           12.0  25.162614
3            500         19.580224  0.037681           12.0  25.535969

```
- There is a clear difference in the total execution time of each output tokens between CPU and GPU.
- The CPU takes more time to generate the required amount of output tokens than GPU.
- As the output tokens limits grew, even with the KV cache, CPU took more time to reach the output limit.
- GPU also followed the same pattern of increased total execution time with increase in output limits.
- Even the ttft or time to first token has a very large gap of time between different runtimes.
- GPU generated a much higher tokens throughput due to their fast execution time.
- In both the runtimes, the total time was dominated by the decoding phase.

Apart from this, on each different output target lengths, the model produced exactly the same response, accroding to the required output lengths. This suggests that the greedy decoding output is very predictable and has low uncertainty.

#### Variable Prompt Lengths
```
Result (CPU): Prefill Phase - 53.88%, Decoding Phase - 46.12%
   prompt_tokens  generated_tokens  total_time_taken       ttft  tokens/s
0           45.0             150.0         42.392357   2.591901  3.538374
1          221.0             150.0         26.667704   8.336147  5.624781
2          529.0             150.0         38.980820  19.278352  3.848046
3         1013.0             150.0         61.277272  39.260383  2.447890
4         2025.0             150.0        101.147778  76.270929  1.482979

Result (GPU): Prefill Phase - 1.77%, Decoding Phase - 98.23%
   prompt_tokens  generated_tokens  total_time_taken      ttft   tokens/s
0           45.0             150.0          6.323349  0.078031  23.721608
1          221.0             150.0          5.780325  0.052616  25.950099
2          529.0             150.0          6.403474  0.089164  23.424783
3         1013.0             150.0          6.089759  0.149646  24.631515
4         2025.0             150.0         13.343222  0.300840  11.241662

```
- The prefill phase in CPU had a increasing trend of `time to first token`, i.e., the model took more time to process longer prompts whereas GPU took less than 1 sec to process different prompt lengths combined.
- As the prompt length increased the CPU spent a lot of time in the prefill phase, while on the GPU, the decoding phase took the highest amount of time.
- Naturally, the GPU had higher tokens throughput than CPU.
- The greedy decoding was consistent. ALthough, it produced a different output for the first prompt, the rest of the prompt had exactly the same output.
  
### Test - 2
#### Temperature, K and P-threshold
In this test, different combinations of temperature, k and probability threshold was used to observe the responses created by the model. The model responses were tested on two different categories:
1. Coding Test - To test if the model response is strict and accurate.
2. Story Writing - To test which combinations are more likely to produced accurate creative responses, without drifting away from the original context.
 
```
Combinations Used
Top-P:
p_threshold = [0.2, 0.5, 0.9]
temp = [0.4, 1.2]

Top-K:
k = [3, 5, 10]
temp = [0.2, 0.6, 1.2]

Decoding Technique Used
1. For Coding Test - Top-P, Top-K
2. For Story Writing - Top-P
```

##### 1. Coding Test
```
Prompt - Write only a Python function that checks whether a string is a palindrome. Do not include explanations.
```

**Top-P**
- As long as the probability threshold or temperature was low, the model produced exactly the same code and was very predictable.
- Moderate probability and higher temperatures still produced correct code, but the model gave explanations as well, even though it was explicitly mentioned in the prompt to not give explanations.
- High probability with lower temperature produced more accurate results than moderate probability and higher temperatures.
- High probability combined with higher tmeperatures produced totally inaccurate and drifting results. The model was prompted to produce a pyhton code, but it produced the palindrome code in C language. The higher proability and temperature resulted in more randomness, hence, the model started producing incoherent outputs.
- In conclusion, low probability and temperature are best for generating accurate codes.

**Top-K**

- Low **top-k (k = 3, 5)** with low or moderate temperatures (**t = 0.2, 0.6**) produced the same correct and consistent Python code.
- Low **top-k** with high temperature (**t = 1.2**) generated correct code but included unnecessary explanations, ignoring the prompt.
- Higher **top-k (k = 10)** with low temperature produced a more detailed but correct solution, along with extra explanations.
- Higher **top-k** with high temperature (**t = 1.2**) produced incorrect and inconsistent code due to increased randomness.
- Overall, lower top-k with low or moderate temperature gives the most accurate and reliable code generation.


##### 2. Story Writing
```
Prompt - Write a short fantasy story about a dragon who is afraid of flying.
```
**Top-P**
- Top-p = 0.2: The model selects from only the highest-probability tokens, producing more predictable and conservative text. The stories remain grammatically correct but frequently become repetitive, with phrases such as "she was being watched" or "she had always been a dragon" repeated multiple times.
- Top-p = 0.5: Increasing top-p introduces greater lexical variety and creativity. However, the outputs still exhibit repetition and occasional inconsistencies, showing a moderate trade-off between diversity and coherence.
- Top-p = 0.9: With a larger candidate token pool, the model generates richer descriptions and more imaginative content. At lower temperature (0.4), the story remains coherent while being more expressive. At higher temperature (1.2), the combination of high top-p and high temperature leads to highly diverse but increasingly incoherent text, including nonsensical words, broken sentences, and loss of narrative structure.

## Final Conclusion

* **Greedy decoding** produced the most deterministic and consistent outputs but offered the least diversity.
* **Temperature, Top-K, and Top-P** controlled the trade-off between accuracy and creativity by varying the randomness in token selection.
* **Lower temperature, Top-K, and Top-P values** generated more accurate, coherent, and instruction-following outputs, making them suitable for coding tasks.
* **Higher sampling values** increased creativity and diversity but also led to repetition, inconsistencies, and incoherent outputs.
* **Top-K and Top-P** generated more diverse responses than greedy decoding, as confirmed by the higher number of unique sentences and Shannon entropy. 
* **GPU inference** significantly outperformed CPU inference with lower latency, faster token generation, and higher throughput. 
* **Overall, decoding parameters should be selected based on the task:** conservative settings for accuracy and reliability, and higher sampling settings for creative text generation. 
