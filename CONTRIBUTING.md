# 🤝 Contributing to WiFi-3D-Fusion

First of all — thanks for taking the time to contribute! 🚀  

This project is open to **ideas, bug reports, and pull requests**.  
Whether you are fixing a typo, improving docs, or experimenting with new features, all contributions are welcome.  

---

## 📋 Table of Contents

- [🛠 How to Contribute](#-how-to-contribute)
- [🐛 Bug Reports](#-bug-reports)
- [🔧 Feature Requests](#-feature-requests)
- [📝 Pull Requests](#-pull-requests)
- [💡 Ideas & Discussions](#-ideas--discussions)
- [🧪 Testing](#-testing)
- [📚 Documentation](#-documentation)
- [✅ Rules](#-rules)
- [🔍 Need help?](#-need-help)

---

## 🛠 How to Contribute

### Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
   ```bash
   git clone https://github.com/MaliosDark/wifi-3d-fusion.git
   cd wifi-3d-fusion
   ```
3. **Set up the environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```
4. **Create a new branch**
   ```bash
   git checkout -b feature/my-idea  # or fix/issue-123
   ```
5. **Make your changes**
6. **Test your changes**
   ```bash
   # Run tests if available
   python test_diagnostics.py
   ```
7. **Commit your changes** with clear messages
   ```bash
   git commit -m "Add: detailed description of your changes"
   ```
8. **Push to your fork**
   ```bash
   git push origin feature/my-idea
   ```
9. **Open a Pull Request** and explain what you did

---

## 🐛 Bug Reports

When filing a bug report, please include:

- **Clear title** that summarizes the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs. **actual behavior**
- **System information**:
  - OS and version
  - Python version
  - Hardware details (especially for WiFi capture devices)
  - Any relevant log output
- **Screenshots** if applicable
- **Possible solution** if you have one in mind

Use the bug report template when creating a new issue.

---

## 🔧 Feature Requests

Have a great idea for WiFi-3D-Fusion? I'd love to hear it! When submitting feature requests:

- **Explain the use case** - what problem would this feature solve?
- **Describe the desired outcome** - what should the feature do?
- **Suggest an implementation** if you have technical ideas
- **Consider compatibility** with existing features
- **Research-focused features** are preferred - remember this is a research tool

---

## 📝 Pull Requests

To ensure your PR is accepted smoothly:

1. **Link related issues** - mention "Fixes #123" or "Relates to #456"
2. **Keep changes focused** - one PR per feature/fix
3. **Maintain code quality**:
   - Follow Python style guidelines (PEP 8)
   - Include comments for complex logic
   - Use descriptive variable names
4. **Update documentation** if necessary
5. **Add tests** for new functionality
6. **Ensure all tests pass**
7. **Describe your changes** in detail in the PR description

PR review process:
- I'll review PRs as soon as possible
- You might be asked to make changes
- Changes will be merged when they meet quality standards

---

## 💡 Ideas & Discussions

Not all contributions need to be code. You can also:

- **Start a Discussion** about research directions or applications
- **Share your experiments** with the system
- **Suggest improvements** to the documentation
- **Ask questions** about implementation details
- **Provide feedback** on usability

For ideas that aren't ready for code yet:
- Open a **Discussion** in the GitHub repository
- Tag your discussion appropriately (Question, Idea, etc.)
- Provide as much detail as possible

---

## 🧪 Testing

Testing is crucial for maintaining a stable codebase:

- **Run existing tests** before submitting changes
  ```bash
  python test_diagnostics.py
  ```
- **Add new tests** for new functionality
- **Test on different devices** if possible (especially for CSI capture)
- **Test with different WiFi devices** where applicable
- **Document any special testing setups**

---

## 📚 Documentation

Good documentation makes the project more accessible:

- **Update README.md** when adding major features
- **Document code** with docstrings and comments
- **Create examples** for new features
- **Update diagrams** if you change system architecture
- **Improve installation instructions** if needed

---

## ✅ Rules

To maintain a productive environment:

- **Keep all interactions constructive and respectful**
- **Follow the [Code of Conduct](CODE_OF_CONDUCT.md)**
- **Respect privacy concerns** inherent to WiFi sensing
- **Remember this project is research-only** - no surveillance/spy features
- **Give proper attribution** when using others' work
- **Be patient** with review processes
- **Accept feedback graciously**

### Ethical Guidelines

Given the nature of WiFi sensing:

- **Consider privacy implications** of any feature you add
- **Default to privacy-preserving options**
- **Avoid features that could enable covert surveillance**
- **Be transparent** about system capabilities

---

## 🔍 Need help?

If you're unsure where to start:

- **Check open issues** - especially those labeled "good first issue"
- **Read the documentation** thoroughly
- **Ask in Discussions** if something is unclear
- **Check the wiki** (if available) for detailed guides
- **Review past PRs** to understand the codebase better

### Communication Channels

- **GitHub Issues** for bugs and feature requests
- **GitHub Discussions** for questions and ideas
- **Pull Requests** for code contributions

---

## 🙏 Thank You

Your contributions help make WiFi-3D-Fusion better for researchers and developers working with WiFi sensing technology. Every contribution, no matter how small, is valuable and appreciated!

---

*Last updated: August 26, 2025*
