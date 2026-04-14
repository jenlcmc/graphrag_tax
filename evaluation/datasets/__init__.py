"""Dataset registry.

Add new datasets here so run_eval.py can discover them automatically.
To add a new dataset:
  1. Create evaluation/datasets/<name>.py implementing Dataset.
  2. Import the class below and add it to REGISTRY.
"""

from evaluation.datasets.taxbench import TaxBenchDataset
from evaluation.datasets.irs_form_qa import IRSFormQADataset
from evaluation.datasets.sara_v3 import SARAV3Dataset

# Maps --dataset flag value to dataset class.
REGISTRY: dict[str, type] = {
    "taxbench":   TaxBenchDataset,
    "irs_form_qa": IRSFormQADataset,
  "sara_v3": SARAV3Dataset,
}
