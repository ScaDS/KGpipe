# Docs (MkDocs)

This repository uses **MkDocs + Material** to build the documentation site from the Markdown files in `docs/`.

## Local preview

### Option A: pip

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

Then open the URL shown in the terminal (usually `http://127.0.0.1:8000/`).

### Option B: uv (recommended if you use uv)

```bash
uv pip install -e ".[docs]"
mkdocs serve
```

## Build

```bash
mkdocs build --strict
```

The static site is written to `site/`.

## Navigation / sidebar

Edit `mkdocs.yml` (`nav:` section) to control:
- sidebar structure
- ordering
- page titles

## Deployment (GitHub Pages)

Deployment is handled by the GitHub Actions workflow:
- `.github/workflows/docs.yml`

In your GitHub repo settings, set:
- **Settings → Pages → Source**: **GitHub Actions**

