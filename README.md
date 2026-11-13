# cigAMT project website

Website source for **cigAMT: A Massive-Scale Benchmark Dataset and Baselines for Audio Magnetotelluric Signal Understanding and Restoration**.

Target repository: `jackzhangqing/cigAMT`  
Target GitHub Pages address, after deployment: <https://jackzhangqing.github.io/cigAMT/>

## Release status

| Resource | Status |
| --- | --- |
| Project website | Static source prepared in `docs/`; deployment requires the target GitHub account. |
| Benchmark training and evaluation code | Release preparation; the original experiment repositories are not included in this website repository. |
| Dataset | A separate local sample package and data card have been prepared. No Hugging Face upload is performed by this setup. |
| Models and checkpoints | Not included; no download link is advertised as available before publication. |
| Paper and supplementary material | Their public links are configured when approved versions are available. |

This repository keeps the website, benchmark-code release, and dataset release distinct. The original data and experiment code are read only during preparation. Assets used on the page are copies of the authors' manuscript figures.

## Local preview

From this repository's directory:

```bash
python preview.py
```

The helper opens the local page in your default browser. It uses port 8873, or another free port if that port is occupied. Press Ctrl+C in its terminal to stop. Use `python preview.py --no-browser` to print the URL without opening a browser. Using a local HTTP server allows the page to load its JSON configuration and results. Opening `index.html` directly as a local file is not the supported preview method.

## Files to edit

| Path | Purpose |
| --- | --- |
| `docs/index.html` | Page content and section structure. |
| `docs/site-config.json` | Public paper, repository, dataset, and related resource URLs. |
| `docs/assets/` | Copied paper figures and website assets. |
| `.github/workflows/pages.yml` | GitHub Pages deployment; the uploaded artifact contains **only `docs/`**. |
| `code/README.md` | Scope and preparation status of the four original code modules. |
| `DEPLOYMENT_ZH.txt` | Chinese deployment instructions. |
| `preview.py` | Local HTTP preview helper, using only Python's standard library. |
| `tools/check_site.py` | Read-only validation of local links, counts, scientific result structure, and copied-image checksums. |

When a resource is not yet public, keep its URL `null` in `docs/site-config.json`. The current page enables links through `paperUrl`, `datasetUrl`, and `repositoryUrl` with `repositoryPublished`. Set `repositoryPublished` to `true` only after the target repository exists. `projectUrl` describes the intended website address. The remaining fields (`supplementaryUrl`, `benchmarkCodeUrl`, `sampleDownloadUrl`, and `releaseStatus`) are reserved for later release updates; they do not currently create page links. Update the visible availability text in `docs/index.html` and `docs/guide.html` when a release occurs. Do not replace pending downloads with unrelated repositories or personal pages.

The local sample preparation directory is kept outside this repository. Uploading the website does not upload that package or any original data. The `.gitignore` also excludes common private-release directory names and scientific-array formats. Avoid adding data files, checkpoints, credentials, local absolute paths, or unpublished result exports to `docs/`.

## Deploy a new GitHub Pages repository

1. Sign in to GitHub as **jackzhangqing** and create an empty repository named **cigAMT**. For GitHub Free, use a public repository. Do not initialize the new repository with a separate README, license, or `.gitignore` if you will use the commands below.
2. In this independent website directory, upload the website source to the new repository. A first-time Git upload can use:

   ```bash
   git init -b main
   git add docs .github/workflows/pages.yml README.md DEPLOYMENT_ZH.txt .gitignore code/README.md asset-manifest.json preview.py tools
   git commit -m "Add cigAMT project website"
   git remote add origin https://github.com/jackzhangqing/cigAMT.git
   git push -u origin main
   ```

   Use GitHub's normal browser or credential-manager sign-in. Do not put a password or access token in the remote URL, scripts, or committed files. The commands apply to an empty new repository; check the existing Git state before repeating them.
3. Open the repository's **Settings → Pages**. Under **Build and deployment**, set **Source** to **GitHub Actions**. This project uses an Actions workflow, rather than branch-based `/docs` publishing.
4. Open **Actions → Deploy project website**. If the first push happened before Pages was enabled, select **Run workflow** on `main`.
5. After the build and deployment jobs succeed, open the URL shown by the `github-pages` deployment. The expected project URL is <https://jackzhangqing.github.io/cigAMT/>.

Later pushes to `main` affecting `docs/`, the workflow, the validation script, or the asset manifest trigger validation and deployment. Changes only to this README do not need a site deployment. No Node installation or frontend build is required.

Validate the source without writing to the research directories:

```bash
python tools/check_site.py
```

Official setup reference: [Using custom workflows with GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Design references

The page organization was informed by the examples supplied by the authors:

- [FISBe](https://kainmueller-lab.github.io/fisbe/index#metrics): dataset documentation, evaluation metrics, and benchmark results.
- [MicroVQA](https://jmhb0.github.io/microvqa/): a concise paper introduction, task examples, and research-resource entry points.
- [IMIS-Bench](https://github.com/uni-medical/IMIS-Bench): separation of dataset access, benchmark code, and usage instructions.

The website implementation is written for cigAMT; it does not copy those projects' templates or scientific figures.

## Licensing

No code or dataset license has been selected on the authors' behalf. Licensing for the website, benchmark code, data, and third-party assets must be confirmed before their public release. The absence of a license here should not be read as permission to redistribute the original experiment repositories or data.
