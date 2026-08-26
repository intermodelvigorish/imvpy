# Releasing IMVpy

IMVpy is published on package indexes under the distribution name `imvpy`; users
also import it as `imvpy`. Versions follow Semantic Versioning and PyPI releases
are immutable.

## One-time repository setup

In the repository's Pages settings, choose **GitHub Actions** as the publishing
source. The `docs.yml` workflow then builds the documentation strictly and
deploys it through the `github-pages` environment after each push to `main`.

## One-time trusted publisher setup

Configure pending Trusted Publishers before the first upload. Use these exact
identities:

| Index | Project | Owner | Repository | Workflow | Environment |
|---|---|---|---|---|---|
| TestPyPI | `imvpy` | `intermodelvigorish` | `imvpy` | `publish-test.yml` | `testpypi` |
| PyPI | `imvpy` | `intermodelvigorish` | `imvpy` | `publish.yml` | `pypi` |

Create matching GitHub environments. Require manual approval for the `pypi`
environment. No long-lived PyPI token is required or expected.

Confirm that the `imvpy` name is still available immediately before configuring
the first pending publisher. PyPI project names are global and normalized.

## Prepare a release

1. Choose a version that has never been uploaded to the target index.
2. Update the version in `pyproject.toml`, `src/imvpy/__init__.py`,
   `config/settings.yaml`, and `CITATION.cff`.
3. Move completed `CHANGELOG.md` entries from `Unreleased` into a versioned
   section with the release date.
4. Run all checks listed in `CONTRIBUTING.md` from a clean checkout.
5. Inspect the wheel and source archive; neither may contain datasets,
   credentials, generated outputs, or local paths.
6. Commit the release changes and ensure CI passes on every supported Python
   version.

## TestPyPI

Run the `Publish to TestPyPI` workflow manually. It builds and validates fresh
artifacts before uploading through the `testpypi` GitHub environment. Because
an uploaded filename cannot be replaced, bump the version before retrying a
failed or superseded upload.

Smoke-test the uploaded wheel in an environment where runtime dependencies are
already installed:

```bash
python -m pip install --no-deps --index-url https://test.pypi.org/simple/ imvpy==VERSION
python -c "import imvpy; print(imvpy.__version__)"
```

## PyPI

1. Create a signed tag named `vVERSION` on the validated release commit.
2. Create and publish a GitHub release from that tag using the matching
   changelog section.
3. Approve the protected `pypi` environment deployment.
4. Verify the project metadata and files on
   [PyPI](https://pypi.org/project/imvpy/).
5. Install the exact release in a fresh environment and run a scalar IMV smoke
   test.

The production workflow rejects a GitHub release whose tag does not exactly
match the version embedded in the package metadata.
