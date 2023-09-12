```ruby
# profound_barracuda/tools/model.py
dec_features = self.decoder(
    enc_features[::-1][0],
    enc_features[::-1][1:]
)
```

<br>

Note that the **`enc_features`** list contains all the feature maps starting from the **first encoder block output to the last.**

Therefore, we can **reverse the order** of feature maps in this list: `enc_features[::-1]`

Now the `enc_features[::-1]` list contains the feature map outputs in reverse order (i.e., **from the last to the first** encoder block).

Note that this is important since, on the **decoder** side, we will be utilizing the encoder feature maps starting from the last encoder block output to the first.

Next, we pass the **output of the final** encoder block (i.e., `enc_features[::-1][0]`) and the feature map outputs of all **intermediate** encoder blocks (i.e., `enc_features[::-1][1:]`) to the decoder.

The **output of the decoder** is stored as `dec_features`.
