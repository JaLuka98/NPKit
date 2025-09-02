# template-python-lib

A minimal, modern **Python library template** designed to jumpstart your project with best practices and tooling.

---

## 🚀 Quick start: using this template

### 1. Create a new repository
- Via GitHub UI:
  Click **Use this template** → create `my-awesome-lib`
- Or via CLI:
  ```bash
  gh repo create your-org/my-awesome-lib --template JaLuka98/template-python-lib --public
  ```

### 2. Clone your new repo and enter it
```bash
git clone https://github.com/your-org/my-awesome-lib.git
cd my-awesome-lib
```

### 3. Apply required customisations
Before you start developing, update the following items in your new project:

| Item                        | Where to change                                     | Notes                                           |
|-----------------------------|-----------------------------------------------------|-------------------------------------------------|
| Package name                | `src/template_python_lib/` → `src/your_package/`    | Rename the directory and update all references  |
| Import paths                | All source and test files                           | Replace `template_python_lib` with your package |
| Package metadata            | `pyproject.toml`, `setup.cfg`                       | Update name, author, description, etc.          |
| Version                     | `pyproject.toml`, `setup.cfg`                       | Optionally set initial version                  |
| License                     | `pyproject.toml`, `setup.cfg`, LICENSE              | Choose and update license as needed             |
| README                      | `README.md`                                         | Update project name and description             |
| Documentation               | `docs/`                                             | Update Sphinx config and docstrings             |

### 4. Set up your environment
Set up the necessary packages in an environment of your choice:
```bash
<activate_your_environment>
pip install -e .[dev]
```

### 5. Run tests to verify setup
```bash
pytest
```

---

## 📂 Project structure

```plaintext
.
├── src/
│   └── template_python_lib/       # Your package source code
├── tests/                         # Test suite
├── docs/                          # Sphinx documentation source
├── .github/workflows/             # GitHub Actions CI workflows
├── .pre-commit-config.yaml        # Pre-commit hooks configuration
├── pyproject.toml                 # Build system and tool configurations
├── setup.cfg                     # Package metadata and settings
├── requirements-dev.txt           # Development dependencies
└── README.md                     # This file
```

---

## 🛠️ Local development

- Use the `src` layout to avoid import conflicts.
- Install dev dependencies with `pip install -e .[dev]`.
- Run tests with `pytest`.
- Type-check your code with `mypy`.
- Lint your code with `ruff`.

---

## 🎨 Linting & formatting

- **Ruff** handles linting and formatting checks.
- Run lint checks manually with:
  ```bash
  ruff check src tests
  ```

---

## 🔧 Pre-commit hooks

- Pre-commit hooks run automatically before each commit to catch issues early.
- Set up pre-commit by running:
  ```bash
  pre-commit install
  ```
- Hooks include:
  - Ruff linting
  - Adding trailing whitespaces

---

## 📦 Versioning

- Uses **CalVer** (Calendar Versioning) format: `YYYY.MM.PATCH` (e.g., `2025.09.0`).
- Version is managed with [bump-my-version](https://github.com/callowayproject/bump-my-version).
- Bump versions with:
  ```bash
  bump-my-version patch
  ```
  or with the dedicated GitHub action.

---

## 📚 Documentation

- Documentation is built with **Sphinx**.
- Source files are in the `docs/` directory.
- Ready for hosting on Read the Docs.
- Build docs locally with:
  ```bash
  cd docs
  make html
  ```

---

## ⚙️ Continuous Integration (CI)

- GitHub Actions workflows automate:
  - Linting and type-checking.
  - Running tests on multiple Python versions.
- Ensures code quality and consistency on every push and pull request.

---

## 🤔 Rationale

This template aims to provide a minimal yet modern starting point for Python libraries, incorporating:

- Best practices like the `src` layout to avoid import pitfalls.
- Automated tooling for testing, linting, formatting, and type checking.
- Pre-commit hooks to maintain code quality from the start.
- Clear versioning strategy with CalVer.
- Documentation setup ready for expansion and publishing.

---

## 🤝 Contributing

Contributions are welcome! Please:

- Follow the existing code style and conventions.
- Run tests and linting before submitting PRs.
- Use the pre-commit hooks to catch issues early.
- Update this README as needed.

---

Thank you for using this template — happy coding!
