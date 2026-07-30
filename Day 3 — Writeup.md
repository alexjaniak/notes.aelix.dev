
ORCA: A Distributed Serving System for Transformer-Based Generative Models
https://www.usenix.org/system/files/osdi22-yu.pdf

Transformers are auto-regressive, so the output is fed into the input of the next iteration. The iterations end when the transformer hits it max token cap or an end-of-sequence special token.

Naively just batch the requests together and return 
