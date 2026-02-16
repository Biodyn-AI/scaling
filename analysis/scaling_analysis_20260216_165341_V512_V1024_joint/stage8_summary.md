# Scaling Analysis Summary

- Runs analyzed: 40
- V values: [64, 128, 512, 1024]
- Sizes: ['XXS', 'TINY', 'XS', 'S', 'M', 'L', 'XL']

## Fits

- V=512, x=params, metric=best_val_gauss_nll: alpha=0.1821, c=1.59151, R2=0.8376, entropy_floor_bits=2.2961
- V=1024, x=params, metric=best_val_gauss_nll: alpha=0.0030, c=0, R2=0.0207, entropy_floor_bits=0.0000
- V=512, x=params, metric=best_val_mse: alpha=0.2664, c=1.44376, R2=0.8580, entropy_floor_bits=2.3120
- V=1024, x=params, metric=best_val_mse: alpha=0.0094, c=0, R2=0.0209, entropy_floor_bits=nan
- V=512, x=tokens_seen, metric=best_val_gauss_nll: alpha=0.0374, c=1.18698, R2=0.0000, entropy_floor_bits=1.7125
- V=1024, x=tokens_seen, metric=best_val_gauss_nll: alpha=0.0153, c=0, R2=0.5762, entropy_floor_bits=0.0000
- V=512, x=tokens_seen, metric=best_val_mse: alpha=0.0610, c=1.27476, R2=0.0000, entropy_floor_bits=2.2222
- V=1024, x=tokens_seen, metric=best_val_mse: alpha=0.0489, c=0, R2=0.5835, entropy_floor_bits=nan