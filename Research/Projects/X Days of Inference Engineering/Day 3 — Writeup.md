
ORCA: A Distributed Serving System for Transformer-Based Generative Models
https://www.usenix.org/system/files/osdi22-yu.pdf

Transformers are auto-regressive, so the output is fed into the input of the next iteration. The iterations end when the transformer hits it max token cap or an end-of-sequence special token.

Naively just batch the requests together and return together. The issue is that if one of the requests in the batch finished before others, this naive approach doesn't let you return that request until the entire batch of requests finishes, its latency increases unnecessarily.

Similarly, if a new request comes in, it must wait till the batch is finished entirely. This means that batches of requests require an entirely new instance of the model. 

The solution is iteration-level scheduling. The scheduler selects which requests get executed at what point and manages a "request pool". 

Unfortunately, we can't coalesce tensors of different shapes. We also can't batch the attention operation for requests of different lengths.

You could pad and mask, but then you pay non-insignificant FLOPs. If you batch a 1000-token pre-fill with a 10-token pre-fill, then you do 1000 x 2 work instead of 1010. 

In addition, the attention op requires you to look back at the prefixed sequence, 


