"""Execute with a fresh kernel and preserve outputs even if a cell fails."""
from pathlib import Path
import os
import sys
import json
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

root=Path(__file__).resolve().parent
runtime=root/'.jupyter_runtime'
runtime.mkdir(exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR']=str(runtime)
os.environ['IPYTHONDIR']=str(root/'.ipython')
os.environ['MPLCONFIGDIR']=str(root/'.mplconfig')
os.environ['JUPYTER_PLATFORM_DIRS']='1'
kernel_dir=runtime/'kernels'/'lynx'
kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps({
    'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Python 3 (Lynx research)','language':'python',
    'env':{'MPLCONFIGDIR':str(root/'.mplconfig'),'OPENBLAS_NUM_THREADS':'1',
           'OMP_NUM_THREADS':'1','PYDEVD_DISABLE_FILE_VALIDATION':'1'}}))
os.environ['JUPYTER_PATH']=str(runtime)
path=root/'lynx_momentum_research.ipynb'
nb=nbformat.read(path,as_version=4)
nb.metadata.kernelspec.name='lynx'
def completed(cell,cell_index,**kwargs):
    nbformat.write(nb,path)
    if cell.cell_type=='code':
        print(f'Cell {cell_index} completed: {cell.source.splitlines()[0][:95]}',flush=True)
client=NotebookClient(nb,timeout=1800,kernel_name='lynx',resources={'metadata':{'path':str(root)}},
                      on_cell_executed=completed)
start=time.perf_counter()
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('Executed in',round(time.perf_counter()-start,1),'seconds',flush=True)
