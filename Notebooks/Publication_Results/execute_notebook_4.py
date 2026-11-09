"""Execute the notebook in a fresh kernel, retain outputs and export a browsable HTML copy."""
import os
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

folder=Path(__file__).resolve().parent
path=folder/'Publication_Results_4.ipynb'
os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['JUPYTER_RUNTIME_DIR']=str(folder/'output_4'/'cache'/'jupyter_runtime')
os.environ['IPYTHONDIR']=str(folder/'output_4'/'cache'/'ipython')
os.environ['MPLCONFIGDIR']=str(folder/'output_4'/'cache'/'matplotlib')
os.environ['PYDEVD_DISABLE_FILE_VALIDATION']='1'
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True,exist_ok=True)
book=nbformat.read(path,as_version=4)
started=time.monotonic()

def progress(cell,cell_index,**kwargs):
    if cell.cell_type=='code':
        print(f'Cell {cell_index+1}/{len(book.cells)} at {time.monotonic()-started:.0f}s',flush=True)

if '--in-process' in sys.argv:
    # Fallback for environments with a broken local ZeroMQ kernel. This script
    # itself is launched in a fresh Python process; no existing session is reused.
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from matplotlib_inline.backend_inline import configure_inline_support
    os.chdir(folder)
    shell=InteractiveShell.instance()
    configure_inline_support(shell, 'module://matplotlib_inline.backend_inline')
    count=0
    try:
        for index,cell in enumerate(book.cells):
            if cell.cell_type!='code':continue
            count+=1
            progress(cell,index)
            with capture_output() as captured:
                result=shell.run_cell(cell.source,store_history=True)
            cell.execution_count=count
            cell.outputs=[]
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
            for rich in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
            if result.error_before_exec or result.error_in_exec:
                print(captured.stdout,flush=True)
                result.raise_error()
            nbformat.write(book,path)
    finally:
        nbformat.write(book,path)
else:
    client=NotebookClient(book,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(folder)}},on_cell_start=progress)
    try:
        client.execute()
    finally:
        nbformat.write(book,path)
exporter=HTMLExporter()
body,_=exporter.from_notebook_node(book)
(folder/'output_4'/'Publication_Results_4.html').write_text(body,encoding='utf-8')
print(f'Executed and exported HTML in {time.monotonic()-started:.1f}s',flush=True)
