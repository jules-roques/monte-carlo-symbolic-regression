To install the project:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .  # Install the project in editable mode, important for clean imports
```

To run the validation:

```bash
python scripts/run_validation.py --config configs/your_algo_config.json
```

To run the test:

```bash
python scripts/run_test.py --config configs/your_algo_config.json
```

