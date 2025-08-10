### 1. Set up environment

#### First Time User:
1. conda env create -n mssa -f conda.yaml

#### Subsequent Setup:
1. conda activate mssa
2. source env.sh


### 2. To run Tests
1. pytest

### 3. Visualization
1. python tests/visualize_path.py

File “tests/data/pivot_unit_tests_inputs_outputs.txt” contains inputs and expected outputs for pivoting unit tests. The expected outputs cover all 48 pivots. These inputs for true positive results. Later inputs may test that certain outputs are NOT generated, i.e. they will test against false positive results.
