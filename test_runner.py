# notebook_runner.py
 
import nbformat
import os
import unittest
from nbconvert.preprocessors import ExecutePreprocessor

class TestNotebook(unittest.TestCase):
 
    def test_runner(self):
        nb, errors = self.run_notebook('index5_test_trained_model_simplernn_for_bnc.ipynb')
        self.assertEqual(errors, [])

    def run_notebook(self, notebook_path):
        nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
        dirname = os.path.dirname(notebook_path)
    
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
    
        proc = ExecutePreprocessor(timeout=18000, kernel_name='python3')#5 mins
        proc.allow_errors = True
    
        proc.preprocess(nb, {'metadata': {'path': './'}})
        output_path = os.path.join(dirname, '{}_all_output.ipynb'.format(nb_name))
    
        with open(output_path, mode='wt') as f:
            nbformat.write(nb, f)
    
        errors = []
        for cell in nb.cells:
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if output.output_type == 'error':
                        errors.append(output)
    
        return nb, errors


if __name__ == '__main__':
    unittest.main()