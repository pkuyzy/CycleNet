We follow the implementation from https://github.com/cptq/SignNet-BasisNet. 

# Install Environment

First creating a python 3.7 environment:

```
conda create -n cyclenet python=3.7
conda activate cyclenet
```

For installing torch (details please refer to https://pytorch.org/get-started/previous-versions/):

```
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

For installing torch_geometric (details please refer to https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

```
pip install torch_scatter==2.0.8 torch_sparse==0.6.9 torch_cluster==1.5.9 torch_spline_conv==1.2.1 -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch_geometric==2.0.4
```

For other requirements.

```
pip install -r requirements.txt
```

We also provide the environment which is available in 

You can first generate a new folder:

```
mkdir graphgym
```

Then download the file to this folder through https://drive.google.com/file/d/1DWl8ZBx3UiZSIVS2qjhwi7rud21-NF4z/view?usp=sharing, and run to obtain the environment

```
tar -xzvf graphgym.tar.gz
source ./bin/activate
```



# ZINC

```
cd ./ZINC
```

To run CycleNet (SignNet as the backbone, with the basis invariant cycle encoding):

```
python main_ZINC_graph_regression.py --config ./configs/pna/PNA_ZINC_LapPE_signinv_GIN_mask.json --use_hodge basis
```

To run CycleNet-PEOI (SignNet as the backbone, with the PEOI cycle encoding):

```
python main_ZINC_graph_regression.py --config ./configs/pna/PNA_ZINC_LapPE_signinv_GIN_mask.json --use_hodge PEOI
```

The hodge data generation is available in ./data/molecules.py, the cycle positional encoding is available in ./nets/ZINC_graph_regression/pna_net.py



# CFI graphs and SR graphs

```
cd ./CFI
```

To reproduce the results on the SR graphs:

```
python main_sr.py
```

The key param is "model_name" in line 33, which denotes the backbone model.

We have implemented "CycleNet_Hodge" (which denotes the basis invariant cycle encoding), "CycleNet" (which denotes the PEOI cycle encoding), "SignNet" (which denotes SignNet (Lim et al. 2023)), "PPGN" (which denotes the Provably Power Graph Network (Maron et al. 2019), as powerful as the 3-WL), and "GNN" (which is a plain GNN). 



To reproduce the results on the CFI graphs:

```
python generate_python_file.py
```

The key params include, "k" (line 32), which denotes the k-CFI graph, and "model_name" (line 39) which denotes the backbone model. The backbone model is the same as the one above.



# Betti Number and EPD

```
cd ./Cycle
```

To reproduce the results on predicting the Betti Number

```
python generate_python_file.py
```

The key param is "model_name", which is the same as before.



To reproduce the results on predicting the EPD

```
python run_EPD.py
```

The key param is "model_name", which is the same as before.