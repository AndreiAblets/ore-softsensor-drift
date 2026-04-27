# Data

Download version 1 of the public dataset:

```text
Dataset on ore grinding and grading process
DOI: 10.17632/hgsf7bwkrv.1
URL: https://doi.org/10.17632/hgsf7bwkrv.1
```

Expected local structure:

```text
data/hgsf7bwkrv-1/
  X_src.csv
  Y_src.csv
  X_tar.csv
  Y_tar.csv
```

The Mendeley Data record lists the dataset under CC BY 4.0. The loader uses the released source and target files, removes the all-zero first feature row, and keeps the remaining measured rows.
