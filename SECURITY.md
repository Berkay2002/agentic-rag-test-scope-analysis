# Security Policy

## Academic Research Context

This is a **Master's Thesis research project**, not a production application. While we value security, please understand the limited scope and resources available for security maintenance.

## Supported Versions

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 0.1.x   | :white_check_mark: | Research release |
| < 0.1   | :x:                | Development only |

## Known Limitations

### Research Software

- **No Security Audit**: This code has not undergone professional security review
- **No Production Use**: Not intended for production or critical systems
- **Limited Maintenance**: Security updates depend on thesis timeline
- **Third-Party Dependencies**: Relies on external services (Google AI, Neo4j, PostgreSQL)

### Specific Concerns

1. **API Keys**: Ensure `.env` file is never committed (already in `.gitignore`)
2. **Database Access**: Use strong passwords and restricted network access
3. **LLM Prompts**: User inputs are sent to Google Generative AI
4. **Injection Risks**: While parameterized queries are used, thorough testing is limited
5. **Data Exposure**: Synthetic data only; don't use with real sensitive data

## Reporting a Vulnerability

### What to Report

Report vulnerabilities that could:
- Expose API keys or database credentials
- Allow unauthorized database access
- Cause data corruption or loss
- Enable code injection or execution
- Compromise system integrity

### How to Report

**For security issues, please DO NOT open a public GitHub issue.**

Instead:

1. **Email**: Berkayorhan@hotmail.se or Beror658@student.liu.se
2. **Subject**: "Security: [Brief Description]"
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if applicable)

### What to Expect

- **Acknowledgment**: Within 5 business days
- **Assessment**: Initial assessment within 2 weeks
- **Fix**: Depends on severity and thesis schedule
- **Disclosure**: Coordinate disclosure timeline with you

**Note**: As a thesis project, response times may vary. Critical issues will be prioritized.

## Security Best Practices for Users

### Environment Configuration

```bash
# Use strong, unique passwords
NEO4J_PASSWORD=<strong-random-password>
POSTGRES_PASSWORD=<strong-random-password>

# Restrict API key permissions
GOOGLE_API_KEY=<key-with-minimal-required-scopes>

# Enable tracing only in development
LANGCHAIN_TRACING_V2=false  # in production
```

### Database Security

1. **Neo4j**:
   - Use Neo4j Aura with built-in security
   - Enable SSL/TLS (`neo4j+s://`)
   - Restrict IP access in Aura console
   - Use read-only users where possible

2. **PostgreSQL/Neon**:
   - Always use SSL (`sslmode=require`)
   - Use connection pooling limits
   - Restrict database user permissions
   - Enable row-level security if needed

### Deployment Security

**DO NOT**:
- Deploy to public internet without authentication
- Use in production environments
- Store real sensitive data
- Share API keys or database credentials
- Commit `.env` file to git

**DO**:
- Use environment variables for secrets
- Run in isolated networks
- Keep dependencies updated
- Monitor API usage and costs
- Review third-party service terms

## Dependencies

This project uses third-party services and libraries:

| Dependency | Security Notes |
|------------|----------------|
| Google Generative AI | Review Google's security policies |
| Neo4j | Keep updated, use Aura for managed security |
| PostgreSQL/Neon | Use SSL, keep updated |
| LangChain/LangGraph | Monitor for security updates |
| Python packages | Run `poetry update` regularly |

### Checking for Vulnerabilities

```bash
# Check for known vulnerabilities in dependencies
poetry show --outdated

# Use safety (install separately)
pip install safety
safety check
```

## Data Privacy

- **Synthetic Data Only**: All test data is artificially generated
- **No PII**: No personal information in this research project
- **LLM Processing**: User queries sent to Google AI (review their privacy policy)
- **Local Storage**: Data stored in your configured databases

## Incident Response

In case of a security incident:

1. **Isolate**: Disconnect affected systems
2. **Assess**: Determine scope and impact
3. **Contain**: Revoke compromised credentials
4. **Report**: Contact repository maintainer
5. **Document**: Record timeline and actions taken

## Responsible Disclosure

We appreciate responsible disclosure. If you discover a vulnerability:

- Give us reasonable time to respond before public disclosure
- Don't exploit the vulnerability beyond proof-of-concept
- Don't access or modify others' data
- Follow coordinated disclosure practices

## Academic Ethics

This project follows academic research ethics:

- No malicious code or backdoors
- Transparent about capabilities and limitations
- Honest disclosure of security posture
- Commitment to fixing reported issues (timeline permitting)

## Updates

This security policy will be updated as needed. Check back for changes.

**Last Updated**: November 2024  
**Project Status**: Active thesis research (limited maintenance post-defense)

## Contact

- **Security Issues**: Berkayorhan@hotmail.se (private)
- **General Questions**: GitHub Discussions (public)

---

**Disclaimer**: This is research software. Use at your own risk. See [LICENSE](LICENSE) and [DISCLAIMER.md](DISCLAIMER.md) for complete terms.
