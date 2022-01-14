# Baseline models summary

|              |   Training  |  Validation  |   Testing   | Leader board |   Notes. |
|     :---     |    :----:   |     :---:    |    :---:    |    :---:     |   ---:   |
|   ATAC2GEX   |    0.0017   |    0.0018    |    0.028    |    0.2466    |          |
|   GEX2ATAC   |    0.107    |    0.143     |    0.126    |    0.1616    |   Check  |
|   GEX2ADT    |    0.178    |    0.183     |    0.185    |    0.2618    |          |
|   ADT2GEX    |    0.075    |    0.073     |    0.088    |    0.2635    |          |

## ATAC2GEX:
  * Training dataset: s1d1, s1d2, s1d3, s2d1, s2d4, s2d5, s3d1, s3d6, s3d7
  * Validation dataset: s3d1, s3d6, s3d7
  * Testing: s4d1, s4d8, s4d9
  * Missing: s3d10
  
## GEX2ATAC:
  * Training dataset: s1d1, s1d2, s1d3, s2d1, s2d4, s2d5, s3d1, s3d6, s3d7
  * Validation dataset: s3d1, s3d6, s3d7
  * Testing: s4d1, s4d8, s4d9
  
## GEX2ADT:
  * Training dataset: s1d1, s1d2, s1d3, s2d1, s2d4, s2d5, s3d1, s3d6, s3d7
  * Validation dataset: s3d1, s3d6, s3d7
  * Testing: s4d1, s4d8, s4d9
  
## ADT2GEX:
  * Training dataset: s1d1, s1d2, s1d3, s2d1, s2d4, s2d5, s3d1, s3d6, s3d7
  * Validation dataset: s3d1, s3d6, s3d7
  * Testing: s4d1, s4d8, s4d9
