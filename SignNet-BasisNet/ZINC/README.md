We follow the implementation from https://github.com/cptq/SignNet-BasisNet. 

To run CycleNet (SignNet as the backbone, with the basis invariant cycle encoding):

```
python main_ZINC_graph_regression.py --config ./configs/pna/PNA_ZINC_LapPE_signinv_GIN_mask.json --use_hodge basis
```

To run CycleNet-PEOI (SignNet as the backbone, with the PEOI cycle encoding):

```
python main_ZINC_graph_regression.py --config ./configs/pna/PNA_ZINC_LapPE_signinv_GIN_mask.json --use_hodge PEOI
```

The hodge data generation is available in ./data/molecules.py, the cycle positional encoding is available in ./nets/ZINC_graph_regression/pna_net.py



#### ZINC results

For implementations of SignNet see `layers/deepsigns.py` and to see its use as a positional encoding see `train/train_ZINC_graph_regression.py`. Fo

To run the experiments, use the scripts in `scripts/`.

For example, to run our SignNet on ZINC with PNA base model, use 
```
bash scripts/ZINC/pna/script_ZINC_PNA_signinv_mask.sh
```

Sample results:

| Base Model | Positional Encoding | #eigs | test MAE |
|----------|:---:|:---:|:---:|
| PNA | None | N/A | 0.128 |
| PNA| SignNet | 8 | 0.105 |
| PNA | SignNet | All | 0.084 |

| Base Model | Positional Encoding | #eigs | test MAE |
|----------|:---:|:---:|:---:|
| GatedGCN | None | N/A | 0.252 |
| GatedGCN | SignNet | 8 | 0.121 |
| GatedGCN | SignNet | All | 0.102 |

| Base Model | Positional Encoding | #eigs | test MAE |
|----------|:---:|:---:|:---:|
| Sparse Transformer | None | N/A | 0.283 |
| Sparse Transformer | SignNet | 16 | 0.115 |
| Sparse Transformer | SignNet | All | 0.102 |



### Attribution
Our code is built on top of the [[LSPE repo](https://github.com/vijaydwivedi75/gnn-lspe)] by Dwivedi et al. in 2021, which in turn builds off of the setup in [[this repo](https://github.com/graphdeeplearning/benchmarking-gnns)] from "Benchmarking Graph Neural Networks" by Dwivedi et al. 2020.
