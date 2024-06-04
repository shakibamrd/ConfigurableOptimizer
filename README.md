# LoRA-DARTS

## Installation and Development
First, install the dependencies required for development and testing in your environment.

```
conda create -n confopt python=3.9
conda activate confopt
pip install -e ".[dev, test]"
pip install -e ".[benchmark]"
```

Install the precommit hooks
```
pre-commit install
```

Run the tests
```
pytest tests
```

Run with the slow benchmark tests
```
pytest --benchmark tests
```

Try running an example
```
python examples/searchspace.py
```
