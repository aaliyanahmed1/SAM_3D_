# Contributing to SAM 3D

Thank you for considering contributing to SAM 3D! This document provides guidelines and instructions for contributing.

## 🤝 Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit bug fixes
- ✨ Add new features
- 🧪 Write tests
- 📊 Share use cases

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/SAM_3D.git
cd SAM_3D

# Add upstream remote
git remote add upstream https://github.com/yourusername/SAM_3D.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (runtime + dev + docs)
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Install in development mode
pip install -e .
```

### 3. Create a Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## 📝 Coding Guidelines

### Code Style

We follow PEP 8 and use automated formatters:

```bash
# Format code
black sam3d/ tests/ examples/

# Sort imports
isort sam3d/ tests/ examples/

# Lint
flake8 sam3d/ tests/ examples/

# Type checking
mypy sam3d/
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `SAM3DSegmentor`)
- **Functions/Methods**: `snake_case` (e.g., `segment_with_points`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)
- **Private**: `_leading_underscore` (e.g., `_internal_method`)

### Documentation

All public APIs must have docstrings:

```python
def segment_with_points(
    self,
    image: Union[str, Path, Image.Image, np.ndarray],
    points: List[List[int]],
    labels: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment objects using point prompts.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        points: List of [x, y] coordinates
        labels: List of labels (1=foreground, 0=background)
        
    Returns:
        Tuple of (masks, scores, image_array)
        
    Example:
        >>> segmentor = SAM3DSegmentor()
        >>> masks, scores, img = segmentor.segment_with_points(
        ...     'image.jpg',
        ...     [[100, 100]],
        ...     [1]
        ... )
    """
```

## 🧪 Testing

### Writing Tests

```python
import pytest
from sam3d import SAM3DSegmentor

@pytest.mark.unit
def test_segmentor_initialization():
    """Test basic initialization."""
    segmentor = SAM3DSegmentor(model_type='vit_h')
    assert segmentor.model_type == 'vit_h'

@pytest.fixture
def dummy_image():
    """Create test image fixture."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sam3d --cov-report=html

# Run specific markers
pytest -m unit
pytest -m integration

# Run specific file
pytest tests/test_segmentation.py -v
```

### Test Coverage

- Aim for >80% coverage
- All new features must include tests
- Bug fixes should include regression tests

## 📦 Submitting Changes

### 1. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add text-based segmentation support"

# Or for bug fixes
git commit -m "fix: resolve memory leak in video segmentation"
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

**Example:**
```
feat: add support for SAM 3 text prompts

Implement text-based segmentation using the new SAM 3 model.
This allows users to segment objects using natural language
descriptions instead of points or boxes.

Closes #123
```

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to GitHub and open a pull request
2. Fill in the PR template
3. Link related issues
4. Request review

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No merge conflicts

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Minimal reproducible example

### Bug Report Template

```markdown
**Description**
Clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Initialize segmentor...
2. Call method...
3. See error

**Expected Behavior**
What should happen.

**Environment**
- OS: Ubuntu 22.04
- Python: 3.10
- SAM 3D: 1.0.0
- GPU: NVIDIA RTX 3090

**Code**
\```python
# Minimal reproducible example
\```

**Error Message**
\```
Full error traceback
\```

**Additional Context**
Any other relevant information.
```

## 💡 Requesting Features

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature.

**Motivation**
Why is this feature needed?

**Proposed Solution**
How should it work?

**Alternatives**
Other solutions considered.

**Additional Context**
Examples, mockups, etc.
```

## 📖 Documentation

### Building Docs

```bash
# Install docs dependencies
pip install mkdocs mkdocs-material

# Serve locally
mkdocs serve

# Build
mkdocs build
```

### Documentation Structure

```
docs/
├── installation.md      # Installation guide
├── api_reference.md     # API documentation
├── use_cases.md        # Use cases and examples
└── contributing.md      # This file
```

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤔 Questions?

- 💬 [GitHub Discussions](https://github.com/yourusername/SAM_3D/discussions)
- 📧 Email: dev@sam3d.com
- 💼 LinkedIn: [SAM 3D](https://linkedin.com/company/sam3d)

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone.

### Our Standards

**Positive behaviors:**
- Using welcoming language
- Being respectful
- Gracefully accepting criticism
- Focusing on what's best for the community

**Unacceptable behaviors:**
- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information

### Enforcement

Report violations to dev@sam3d.com. All reports will be reviewed confidentially.

## Thank You! 🙏

Your contributions make SAM 3D better for everyone. We appreciate your time and effort!

