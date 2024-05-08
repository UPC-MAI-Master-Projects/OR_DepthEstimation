
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