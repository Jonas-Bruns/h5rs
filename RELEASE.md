# Releasing h5rs

The first release is `v0.1.0`. The workflow builds a source distribution and
wheels for CPython 3.10–3.15 on Windows x64, Linux x64 and macOS Apple Silicon.
It installs and tests each wheel and validates all package metadata before
publishing.

## First release

1. Create an empty GitHub repository and commit this project to its default
   branch. If Git has no author identity configured, set `user.name` and
   `user.email` locally before committing. Replace `OWNER` and `REPO` below.

   ```sh
   git add .
   git commit -m "Prepare h5rs 0.1.0"
   git branch -M main
   git remote add origin https://github.com/OWNER/REPO.git
   git push -u origin main
   ```

2. Create a GitHub Actions environment called `pypi` under the repository's
   **Settings → Environments**. On PyPI, create a pending Trusted Publisher
   with the following values. This uses GitHub's identity token and requires
   no stored PyPI API token. [PyPI guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)

   | PyPI field | Value |
   | --- | --- |
   | PyPI project name | `h5rs` |
   | Owner | GitHub user or organization |
   | Repository name | GitHub repository name |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

3. Push the version tag. This starts a build with downloadable artifacts.

   ```sh
   git tag -a v0.1.0 -m "h5rs 0.1.0"
   git push origin v0.1.0
   ```

4. After the tag build is green, use the authenticated GitHub CLI to run
   the same workflow on the tag with publishing enabled. It publishes to
   PyPI and then attaches the distributions to the GitHub Release.

   ```sh
   gh workflow run release.yml --ref v0.1.0 -f publish=true
   ```

5. In a clean Python environment, verify the public install.

   ```sh
   python -m pip install h5rs==0.1.0
   python -c "import h5rs; print(h5rs.File)"
   ```

## Subsequent releases

Update the package version in `Cargo.toml`, `crates/h5rs-python/Cargo.toml`
and `pyproject.toml`, and update `Cargo.lock`. Review the bundled notices in
`THIRD_PARTY_LICENSES.txt` when dependencies change. Commit and push the
changes before tagging. Use the corresponding version tag for the build and
publish steps.
