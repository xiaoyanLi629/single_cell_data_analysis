Results comparison

Baseline models summary

|                          |   Training  |  Validation  |   Testing    | Leader board |Predicted features|   Notes   |
|     :---                 |    :----:   |     :---:    |    :---:     |    :---:     |:---:|    ---:   |
|   ATAC2GEX binary input  |    0.0017   |    0.0018    |    0.0270    |    0.2466    |1|           |
|   ATAC2GEX counts input  |    0.0974   |    0.1046    |    0.0709    |    0.2466    |1|           |
|   ATAC2GEX counts normal |    0.098    |    0.1059    |    0.0321    |    0.2466    ||           |
|  xxxxxxxxxxxxxxxxxxxxxx  |    xxxxx    |    xxxxxx    |    xxxxxx    |    xxxxxx    ||  xxxxxxx  |
|          GEX2ADT         |    0.154    |    0.1796    |    0.1837    |    0.2618    ||  baseline         |
|        GEX2ADT PCA       |    0.322    |    0.2859    |    0.4271    |    0.2618    |134|  check         |
|    GEX2ADT batch_effec   |    0.063    |    0.4487    |    0.5586    |    0.2618    |134| one batch train, batch effect |
|    GEX2ADT Seurat inte   |    0.133    |    0.1398    |    0.2169    |    0.2618    |1|           |
|    GEX2ADT latent 50     |    0.188    |    0.1915    |    0.1753    |    0.2618    |1| conver fast, overfitting          |
|    GEX2ADT latent 15     |    0.179    |    0.1853    |    0.1667    |    0.2618    |1|           |
|    GEX2ADT latent 25     |    0.183    |    0.1886    |    0.1677    |    0.2618    |1|           |
|    GEX2ADT latent 10     |    0.185    |    0.1886    |    0.1637    |    0.2618    |1|           |
|    GEX2ADT latent 5      |    0.186    |    0.1895    |    0.1683    |    0.2618    |1|           |
|      GEX2ADT Resnet      |    0.193    |    0.1954    |    0.1892    |    0.2618    |1|  overfitting         |
|   GEX2ADT + Seurat inte  |    0.141    |    0.1469    |    0.1708    |    0.2618    |1| converge fast, robust to overfitting |
|       GEX2ADT CNN        |    0.233    |    0.2046    |    0.2536    |    0.2618    |1|  converge slow        |
|       GEX2ADT GNN        |    xxxxx    |    xxxxxx    |    xxxxxx    |    0.2618    ||           |
| ATAC-ADT2GEX binary norm |    xxxxx    |    xxxxxx    |    xxxxxx    |    xxxxx     |134|           |
