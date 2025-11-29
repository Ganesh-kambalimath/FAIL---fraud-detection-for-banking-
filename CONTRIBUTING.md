# Contributing to Secure Federated Fraud Detection

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages and stack traces

### Suggesting Enhancements

1. Create an issue describing the enhancement
2. Explain why this enhancement would be useful
3. Provide examples of how it would work

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Update documentation as needed
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/secure-federated-fraud-detection.git
cd secure-federated-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

## Code Style

- Follow PEP 8 style guide
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep functions focused and small
- Maximum line length: 100 characters

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    """
    # Implementation
    return True
```

## Testing

- Write tests for all new features
- Ensure existing tests pass
- Aim for >80% code coverage
- Use pytest fixtures for common test setup

```python
def test_example_function():
    """Test example function with various inputs"""
    result = example_function("test", 42)
    assert result is True
```

## Documentation

- Update README.md if adding major features
- Add docstrings to all public functions/classes
- Update API documentation in `docs/`
- Include examples for new functionality

## Commit Messages

Use clear and descriptive commit messages:

```
Add feature: Brief description of what was added

More detailed explanation if needed.
Can span multiple lines.

- Bullet points for specific changes
- Another specific change
```

## Review Process

1. All PRs require at least one review
2. Address reviewer feedback
3. Ensure CI/CD pipeline passes
4. Squash commits before merging if requested

## Priority Areas

We especially welcome contributions in:

- Additional privacy-preserving mechanisms
- Performance optimizations
- Byzantine defense improvements
- Documentation and tutorials
- Test coverage
- Real-world dataset integration
- Deployment and scalability

## Questions?

Feel free to:
- Open an issue for discussion
- Join our community Discord
- Email the maintainers

Thank you for contributing! 🎉
