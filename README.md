# Delta-ML
We leverage the capabilities of the Delta-ML model to predict relevant TADF properties using a lower cost baseline method PPP to obtain comparable accuracy to the QM method ADC(2).

- The model can be trained using the code provided in schnet.py
- Coulson package was used to generate the PPP data
- XYZ coordinates of both the dataset are given as zip files
- XYZ of a few of the molecules from the benchmark INVEST15 dataset that are different from the training set of the pretrained model are used for validating the generalizability of this approach. These are included in the zip file.
