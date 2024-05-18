## Get Data

1. Download the dataset from [here](https://cvcuab-my.sharepoint.com/personal/mmadadi_cvc_uab_cat/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmmadadi%5Fcvc%5Fuab%5Fcat%2FDocuments%2Fcloth3d%2B%2B%5Fsubset%2Ezip&parent=%2Fpersonal%2Fmmadadi%5Fcvc%5Fuab%5Fcat%2FDocuments&ga=1)


## Troubleshooting

```python
ModuleNotFoundError: No module named 'smpl.smpl_np'
```
Solution: 

1. Add paths to the system path
```python
import sys
sys.path.append('cloth3d')
sys.path.append('cloth3d/DataReader')
sys.path.append('cloth3d/DataReader/smpl')

from DataReader.read import DataReader

reader = DataReader()
```
2. Go to [cloth3d/DataReader/read.py](cloth3d/DataReader/read.py) and change the import statement from 
```python
from smpl.smpl_np import SMPLModel
```
to 
```python
from smpl_np import SMPLModel
```