## 🔧 Installation & Setup

### Install `uv` globally  
```sh
pip install uv
```

### Create a virtual environment  
```sh
uv venv create
```

### Create a virtual environment in a specific folder  
```sh
uv venv create .venv
```

### Activate the virtual environment  
```sh
uv venv enter
```

### Remove the virtual environment  
```sh
uv venv destroy
```

---

## 📦 Installing Packages

### Install a package  
```sh
uv pip install PACKAGE
```

### Install from `requirements.txt`  
```sh
uv pip install -r requirements.txt
```

### Install project dependencies from `pyproject.toml`  
```sh
uv pip install .
```

### Install development dependencies  
```sh
uv pip install --dev .
```

---

## 🔄 Managing Dependencies

### Show installed packages  
```sh
uv pip freeze
```

### Remove a package  
```sh
uv pip uninstall PACKAGE
```

### Install all dependencies from `pyproject.toml`  
```sh
uv venv sync
```

### List installed packages  
```sh
uv pip list
```

---

## 🏗️ Working with Requirements

### Generate `requirements.txt` from `pyproject.toml`  
```sh
uv pip compile
```

### Install exactly what's in `requirements.txt`  
```sh
uv pip sync
```

---

## 🏃 Running Scripts in venv

### Run a Python script inside the virtual environment  
```sh
uv venv run python main.py
```

### Run `pytest` inside the virtual environment  
```sh
uv venv run pytest
```

---
