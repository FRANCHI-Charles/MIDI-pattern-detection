# Notes about code

- MorphoOperators : `padding_mode="replicate"` may not be the best option, `'zeros'` should be better for midi.
- array_erosion : not symmetric as in the paper, def as paul
- ERROR in the array definition : torch.Conv2D apply cross-correlation, not convolution => we have to transpose the kernel
- TO MODIFY : midi.py -> %3 !=0 [...] must depend on tatum