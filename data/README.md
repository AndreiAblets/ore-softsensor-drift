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

The CSV files are not included here. The Mendeley Data record lists the dataset under CC BY 4.0.

The public metadata do not define the ore type, mill type, class semantics, sampling rate, or source/target split criterion. The reproduction code uses the numeric labels and the released source/target files as provided, removes the all-zero first feature row, and keeps the remaining eight measured rows.
