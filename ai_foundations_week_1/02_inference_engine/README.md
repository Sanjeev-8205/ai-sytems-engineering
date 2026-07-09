# Objective
To understand the implementation of the entire text generation pipeline after the token ids and the embeddings are computed.

---
# Models
1. Model - `Qwen/Qwen2.5-0.5B`
2. Tokenizer - `Qwen/Qwen2.5-0.5B`
3. Model for Causal LM - `Qwen/Qwen2.5-0.5B`

---
# Experiments
## Logits and Softmax
Create a fake logits dictionary with tokens and logit scores mapped to the tokens. Create a softmax function using numpy and use it on these fake logits to convert them to probabilities. Change few logits and observe how it affects the probability distribution.

## Decoding Strategies
1. **Temperature**
   - Create a temperature softmax function where probabilities varies with diffferent temperature. Use the json file - `fake_vocab.json`, which has fake tokens and logits to compute probabilities from this temperature softmax function. Create graph from this computed probabilities showing how the probability distribution varies with different temperature settings.

2. **Greedy**
   - Create a argmax function and use an array of random values to check if it returns the index of the maximum value in the array. Verify it by comparing it with built-in numpy or torch argmax function.
   - Autoregressive Generation: Use the json file `fake_transformer_1000.json` to simulate a fake transformer text generation. 
3. **Top-k**
   - Use the `fake_vocab.json` file to compute the top-10 probabilities and map it back to the token.
   - Autoregressive Generation: Simulate a fake transformer using the same `fake_transformer_1000.json` used in the greedy method.
4. **Top-p**
   - Try different probability threshold to observe the number of tokens selected, revealing the probability distribution.
   - Autoregressive Generation: Simulate a fake transformer using the same `fake_transformer_1000.json` used in the greedy method.

## AutoRegressive Generation Using Fake Tokens and Logits
Used `Qwen/Qwen2.5-0.5B` to create a AutoRegressive Generation of tokens with the help of `fake_ transformer_1000.json` file.

- Used 10 different starting words, k=3 and p=0.95.
- In each `for-loop`, all the different techniques receives the same starting prompt, and they ran inside a while loop until the end of sequence is reached. One `for-loop` is completed after all the decoding techniques reached end of sequence.
- This was simulated for a total of 1000 loops.
- Three different categories of results were produced: Unique Sentences, Most Frequent Senteces and Shannon's Capacity.
  
Note: The AutoRegressive Generation for all decoding technique is carried out in a single function, but seperate implementation is possible as well.
Here is the mermaid diagram of the AutoRegressive Generation:
```mermaid
flowchart TD

A([Start]) --> B[Initialize store_outputs, k, p, sizes]
B --> C[Repeat 1000 iterations]
C --> D[Choose random starting prompt]

D --> G1
D --> T1
D --> K1
D --> P1

%% Greedy
subgraph Greedy_Decoding
G1[Copy prompt] --> G2{Prompt exists?}
G2 -- No --> G5[Finished]
G2 -- Yes --> G3[Compute logits and softmax]
G3 --> G4[Select highest probability token]
G4 --> G6[Append token]
G6 --> G2
end

%% Temperature
subgraph Temperature_Decoding
T1[Copy prompt] --> T2{Prompt exists?}
T2 -- No --> T5[Finished]
T2 -- Yes --> T3[Apply temperature softmax T=0.6]
T3 --> T4[Sample next token]
T4 --> T6[Append token]
T6 --> T2
end

%% Top-k
subgraph TopK_Decoding
K1[Copy prompt] --> K2{Prompt exists?}
K2 -- No --> K5[Finished]
K2 -- Yes --> K3[Compute softmax and keep top k tokens]
K3 --> K4[Renormalize probabilities and sample]
K4 --> K6[Append token]
K6 --> K2
end

%% Top-p
subgraph TopP_Decoding
P1[Copy prompt] --> P2{Prompt exists?}
P2 -- No --> P8[Finished]
P2 -- Yes --> P3[Compute softmax and sort probabilities]
P3 --> P4[Keep tokens with cumulative probability less than or equal to p]
P4 --> P5[Include first token exceeding p if needed]
P5 --> P6[Renormalize probabilities and sample]
P6 --> P7[Append token]
P7 --> P2
end

G5 --> S
T5 --> S
K5 --> S
P8 --> S

S[Store completed sentence counts] --> L{1000 iterations complete?}

L -- No --> D
L -- Yes --> M[Count unique sentences]
M --> N[Find most frequent sentence]
N --> O[Compute entropy for each decoding strategy]
O --> Z([End])
```

## Real Inference
1. Load the `Qwem/Qwen2.5-0.5B`model, tokenizer and model for causal lm.
2. Build a real inference pipeline function using this model from the prompt tokenization to the continuos text generation, with all the decoding techniques.
3. Track the following metrics - total time, tokens/sec, prompt length and oenerated tokens for each complete execution.
4. Test the `greedy` decoding strategy with variable output token target and variable prompt lengths using the pipeline function. Run it on both CPU and GPU and obesrve the difference in the latency, token/sec.
5. Carry out two tests: `coding test` - to fact check the model response and `story test` - to test the creativeness of the model. Observe which pair of temperature and p produces the best results for each test. Use `Top-p` decoding technique with temperature for this test, i.e., test different pairs of p and temperature to test the response of the model.

Here is the mermaid diagram of the inference pipeline:

```mermaid
flowchart TD

A([Start]) --> B[Initialize Device, Model, Tokenizer, Parameters]
B --> C[Tokenize Prompt]
C --> D{Select Decoding Strategy}

D -->|Greedy| G
D -->|Temperature| T
D -->|Top-K| K
D -->|Top-P| P
D -->|Invalid| X[Print Invalid Strategy]

%% ---------------- Greedy ----------------

subgraph Greedy_Decoding
G[Start Timer]

G --> G1[Forward Pass<br/>Hidden States<br/>Logits<br/>Past Key Values]
G1 --> G2[Scale Logits]
G2 --> G3[Softmax]
G3 --> G4[Select Highest Probability Token]
G4 --> G5[Append Token]

G5 --> G6{First Token?}
G6 -->|Yes| G7[Record TTFT]
G6 -->|No| G8

G7 --> G8
G8{EOS or Max Tokens?}
G8 -->|No| G1
end
G8 -->|Yes| R

%% ---------------- Temperature ----------------

subgraph Temperature_Decoding
T[Start Timer]

T --> T1[Forward Pass<br/>Hidden States<br/>Logits<br/>Past Key Values]
T1 --> T2[Scale Logits by Temperature]
T2 --> T3[Softmax]
T3 --> T4[Random Sampling]
T4 --> T5[Append Token]

T5 --> T6{First Token?}
T6 -->|Yes| T7[Record TTFT]
T6 -->|No| T8

T7 --> T8
T8{EOS or Max Tokens?}
T8 -->|No| T1
end
T8 -->|Yes| R

%% ---------------- Top-K ----------------

subgraph TopK_Decoding
K[Start Timer]

K --> K1[Forward Pass<br/>Hidden States<br/>Logits<br/>Past Key Values]
K1 --> K2[Scale Logits]
K2 --> K3[Keep Top-K Logits]
K3 --> K4[Softmax]
K4 --> K5[Random Sampling]
K5 --> K6[Append Token]

K6 --> K7{First Token?}
K7 -->|Yes| K8[Record TTFT]
K7 -->|No| K9

K8 --> K9
K9{EOS or Max Tokens?}
K9 -->|No| K1
end
K9 -->|Yes| R

%% ---------------- Top-P ----------------

subgraph TopP_Decoding
P[Start Timer]

P --> P1[Forward Pass<br/>Hidden States<br/>Logits<br/>Past Key Values]
P1 --> P2[Scale Logits]
P2 --> P3[Sort Logits]
P3 --> P4[Keep Top-P Probabilities]
P4 --> P5[Random Sampling]
P5 --> P6[Append Token]

P6 --> P7{First Token?}
P7 -->|Yes| P8[Record TTFT]
P7 -->|No| P9

P8 --> P9
P9{EOS or Max Tokens?}
P9 -->|No| P1
end
P9 -->|Yes| R

R["Return Metrics<br/><br/>• TTFT<br/>• Total Time<br/>• Prompt Tokens<br/>• Generated Tokens"]
```