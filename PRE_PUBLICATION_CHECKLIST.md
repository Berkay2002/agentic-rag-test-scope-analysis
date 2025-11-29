# Pre-Publication Checklist

This checklist helps ensure your repository is ready for public release.

## ✅ Completed Items

- [x] **LICENSE**: Academic Research License with proper attribution
- [x] **README.md**: Updated with thesis context, Mermaid diagrams, and badges
- [x] **CITATION.cff**: GitHub citation file for easy academic citation
- [x] **CONTRIBUTING.md**: Clear guidelines for academic collaboration
- [x] **DISCLAIMER.md**: Usage restrictions and limitations
- [x] **SECURITY.md**: Security policy and responsible disclosure
- [x] **pyproject.toml**: Updated metadata and classifiers
- [x] **GitHub Templates**: Issue templates and PR template
- [x] **GitHub Actions**: Basic CI workflow for code quality
- [x] **TODO Comments**: Removed or converted to notes
- [x] **Personal Information**: Updated with your details

## 🔍 Before Making Public

### 1. Sensitive Information Review

- [ ] **No API keys in code**: Search for `GOOGLE_API_KEY`, `LANGCHAIN_API_KEY`, etc.
- [ ] **No passwords**: Check for hardcoded passwords or credentials
- [ ] **No personal tokens**: Verify no GitHub tokens, Neo4j passwords, etc.
- [ ] **.env file not tracked**: Ensure `.env` is in `.gitignore` and not committed
- [ ] **No proprietary data**: Confirm only synthetic data is included

```bash
# Run these checks:
git ls-files | xargs grep -l "API_KEY\|PASSWORD\|SECRET\|TOKEN" 2>/dev/null
git status --ignored | grep ".env"
```

### 2. Repository Settings (on GitHub)

- [ ] **Repository name**: Choose appropriate name (e.g., `agentic-graphrag-test-analysis`)
- [ ] **Description**: Add clear description
- [ ] **Topics/Tags**: Add relevant tags (machine-learning, knowledge-graph, thesis, etc.)
- [ ] **License**: Select "Other" (using custom academic license)
- [ ] **README preview**: Verify it renders correctly
- [ ] **Discussions enabled**: Enable GitHub Discussions for Q&A
- [ ] **Issues enabled**: Enable Issues for bug reports
- [ ] **Branch protection**: Consider protecting main branch

### 3. Documentation Review

- [ ] **README links work**: Test all hyperlinks
- [ ] **Mermaid diagrams render**: Verify on GitHub
- [ ] **Code examples accurate**: Test example commands
- [ ] **Installation steps clear**: Follow setup instructions fresh
- [ ] **Citation format correct**: Verify BibTeX and APA formats

### 4. Code Quality

```bash
# Run these before committing:
poetry run black src/ tests/
poetry run ruff check src/ tests/
poetry run pytest  # if you have tests
```

- [ ] **Code formatted**: Run Black
- [ ] **Linting passes**: Run Ruff
- [ ] **Tests pass**: If applicable
- [ ] **Dependencies updated**: Run `poetry update` if needed

### 5. Content to Update/Add

#### Update These with Your Repository URL

Once you create the GitHub repository, replace `yourusername` in:

- [ ] `README.md`: All GitHub URLs
- [ ] `CITATION.cff`: repository-code field
- [ ] `pyproject.toml`: repository and homepage fields
- [ ] `CONTRIBUTING.md`: Issue/Discussion URLs
- [ ] `.github/ISSUE_TEMPLATE/*.md`: Repository references

```bash
# After creating repo, run:
GITHUB_USER="yourusername"
REPO_NAME="your-repo-name"
find . -type f -name "*.md" -o -name "*.toml" -o -name "*.cff" | \
  xargs sed -i '' "s|yourusername/agentic-rag-test-scope-analysis|$GITHUB_USER/$REPO_NAME|g"
```

#### Optional: Add These Files

- [ ] **CHANGELOG.md**: Track version changes
- [ ] **.github/FUNDING.yml**: If accepting sponsorships
- [ ] **CODE_OF_CONDUCT.md**: Community standards (we can add this)
- [ ] **docs/** folder: Extended documentation
- [ ] **examples/** folder: Usage examples

### 6. GitHub Repository Setup

1. **Create repository on GitHub**:
   ```bash
   # Using GitHub CLI
   gh repo create your-repo-name --public --source=. --remote=origin
   
   # Or manually on github.com
   ```

2. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

3. **Configure repository settings**:
   - Add description
   - Add topics: `knowledge-graph`, `rag`, `langgraph`, `neo4j`, `thesis`, `research`
   - Enable Discussions
   - Enable Issues
   - Add About section

4. **Verify**:
   - Check README renders correctly
   - Test Mermaid diagrams display
   - Verify badges work
   - Test citation feature (Cite this repository button)

### 7. Final Pre-Flight Checks

```bash
# Clean build artifacts
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Check what will be committed
git status

# Review changes
git diff

# Check for large files
find . -type f -size +1M -not -path "./.*"
```

- [ ] **No large files**: Verify no accidentally large files (>1MB)
- [ ] **No build artifacts**: Clean `__pycache__`, `*.pyc`, etc.
- [ ] **.gitignore comprehensive**: Verify all necessary exclusions
- [ ] **Commit history clean**: No sensitive information in history

### 8. Post-Publication

After making repository public:

- [ ] **Test clone**: Clone fresh copy and verify setup works
- [ ] **Share with supervisor**: Get feedback before wider sharing
- [ ] **Monitor issues**: Set up notifications
- [ ] **Update LinkedIn/CV**: Link to repository
- [ ] **LiU thesis archive**: Coordinate with university requirements

## 📝 Notes

### Things to Consider

1. **GitHub Actions CI**: The workflow will run on push/PR. Ensure it passes.
2. **LangSmith Tracing**: Keep disabled in CI (uses your API key)
3. **Database Credentials**: Never commit real database passwords
4. **Cost Awareness**: Mention API costs in README (done ✓)

### After Thesis Defense

- Update README with thesis results/findings
- Add link to published thesis (if available)
- Archive repository or mark as complete
- Consider writing blog post about the work

## ❓ Questions to Answer

Before going public, ensure you can answer:

- [ ] **Why public?** Academic transparency, portfolio, reproducibility
- [ ] **Who is audience?** Researchers, students, thesis committee
- [ ] **What can they do?** Study, cite, learn from (not commercial use)
- [ ] **Support expectations?** Limited, best-effort during thesis period
- [ ] **Ericsson approval?** Ensure you have permission for public release

## 🚀 Ready to Publish?

When all items are checked:

```bash
# Final commit
git add .
git commit -m "chore: prepare repository for public release"
git push origin main

# Tag release
git tag -a v0.1.0 -m "Initial public release - Research implementation"
git push origin v0.1.0
```

---

**Contact**: Berkay Orhan (Berkayorhan@hotmail.se)  
**University**: Linköping University  
**Last Updated**: November 28, 2025
