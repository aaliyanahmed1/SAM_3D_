# SAM 3D Tests

Comprehensive test suite for SAM 3D with unit, integration, and performance tests.

## Test Structure

```
tests/
├── test_segmentation.py      # Segmentation module tests
├── test_reconstruction.py    # 3D reconstruction tests
├── test_integration.py       # Integration/workflow tests
├── test_performance.py       # Performance benchmarks
├── test_stress.py           # Stress tests
└── README.md                # This file
```

## Running Tests

### All Tests

```bash
pytest
```

### Unit Tests Only

```bash
pytest -m unit
```

### Integration Tests

```bash
pytest -m integration
```

### With Coverage

```bash
pytest --cov=sam3d --cov-report=html
```

### Specific Test File

```bash
pytest tests/test_segmentation.py -v
```

## Test Markers

Tests are marked with the following markers:

- `unit`: Fast unit tests (no model required)
- `integration`: Integration tests (may require models)
- `slow`: Slow-running tests
- `gpu`: Tests requiring GPU
- `api`: API endpoint tests

### Running Specific Markers

```bash
# Run only unit tests
pytest -m unit

# Run everything except slow tests
pytest -m "not slow"

# Run GPU tests
pytest -m gpu
```

## Test Configuration

Configuration in `pytest.ini`:

```ini
[pytest]
testpaths = tests
markers =
    unit: Fast unit tests
    integration: Integration tests
    slow: Slow tests
    gpu: GPU required
    api: API tests
```

## Writing Tests

### Unit Test Example

```python
import pytest
from sam3d.segmentation import SAM3DSegmentor

@pytest.mark.unit
def test_segmentor_initialization():
    """Test basic initialization."""
    segmentor = SAM3DSegmentor(model_type='vit_h')
    assert segmentor.model_type == 'vit_h'
```

### Integration Test Example

```python
@pytest.mark.integration
@pytest.mark.skip(reason="Requires model checkpoint")
def test_full_pipeline():
    """Test complete workflow."""
    segmentor = SAM3DSegmentor()
    segmentor.load_model('checkpoint.pth')
    # ... test logic
```

### Fixtures

```python
@pytest.fixture
def dummy_image():
    """Create test image."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

def test_with_fixture(dummy_image):
    """Use fixture in test."""
    assert dummy_image.shape == (512, 512, 3)
```

## Coverage Reports

After running tests with coverage:

```bash
pytest --cov=sam3d --cov-report=html
```

Open coverage report:

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Continuous Integration

Tests run automatically on:
- Every push to main/develop
- Every pull request
- Nightly builds

See `.github/workflows/ci.yml` for CI configuration.

## Test Data

For tests requiring data files:

1. Create `tests/data/` directory
2. Add small test images/videos
3. Add to `.gitignore` if files are large

Example:
```
tests/
├── data/
│   ├── test_image.jpg
│   ├── test_video.mp4
│   └── checkpoints/
│       └── sam_vit_b.pth (small model for testing)
```

## Performance Testing

Run performance benchmarks:

```bash
pytest tests/test_performance.py --benchmark-only
```

View benchmark results:

```bash
pytest tests/test_performance.py --benchmark-only --benchmark-compare
```

## Troubleshooting

### Tests fail with "Model not found"

```bash
# Download test checkpoint
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
```

### CUDA out of memory in GPU tests

```bash
# Run with CPU only
pytest -m "not gpu"
```

### Import errors

```bash
# Install package in development mode
pip install -e .
```

## Contributing Tests

When contributing new features:

1. Add unit tests for core functionality
2. Add integration tests for workflows
3. Ensure coverage > 80%
4. Run full test suite before PR

```bash
# Check your tests
pytest tests/ -v
pytest --cov=sam3d --cov-report=term-missing
```

