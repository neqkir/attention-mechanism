Learning resources and gathering on the concept of attention-mechanism 

(1) Attention mechanism in **Attention is all you need** https://arxiv.org/abs/1706.03762 (transformers)

" An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

We call our particular attention “Scaled Dot-Product Attention”.   The input consists of queries and keys of dimension <img src="https://render.githubusercontent.com/render/math?math=d_k">, and values of dimension <img src="https://render.githubusercontent.com/render/math?math=d_v">.
We compute the dot products of the query with all keys, divide each by
<img src="https://render.githubusercontent.com/render/math?math=\sqrt{d_k}">, and apply a softmax function to obtain the weights on the values. "

<img src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_33_0.png" width=10% height=10%>

```
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
 ```

" The Transformer uses multi-head attention in three different ways: 

1) In “encoder-decoder attention” layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as (cite).

2) The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.

3) Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot- product attention by masking out (setting to 
<img src="https://render.githubusercontent.com/render/math?math=-\infty">) all values in the input of the softmax which correspond to illegal connections." 
