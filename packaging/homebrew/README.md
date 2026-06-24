# Homebrew distribution

Sotto ships via a **Homebrew tap** (a separate repo, `tomoima525/homebrew-sotto`).
Two delivery formats are possible:

- **Formula** (this doc) — installs the `sotto` CLI/menu-bar app into a Python
  virtualenv. No code signing required. Install with
  `brew install tomoima525/sotto/sotto`, then `sotto run`.
- **Cask** (future) — a notarized `Sotto.app` for a clickable, Login-Item
  experience. Requires an Apple Developer ID for signing + notarization.

Apple Silicon only — MLX has no Intel build (`depends_on arch: :arm64`).
`sounddevice` bundles its own PortAudio, so there is no system audio dependency.
The ~4 GB models are **not** bundled; they download on first run (`sotto download`).

> The notes below are not the textbook Homebrew-Python flow — they're what
> actually works for this MLX/ML dependency tree, after the textbook flow
> (`brew update-python-resources` + `virtualenv_install_with_resources`) failed.
> Both pitfalls and their fixes are documented so the next bump is mechanical.

## Releasing a version the formula can point at

The formula's `url` is a tagged GitHub source tarball, so each formula update
needs a matching tag.

1. Bump `version` in `pyproject.toml`. `sotto.__version__` is derived from the
   installed package metadata, so it follows automatically.
2. Merge to `main`, then tag and release:
   ```sh
   git tag 1.0.2 && git push origin 1.0.2
   gh release create 1.0.2 --generate-notes
   ```
3. Get the tarball sha256. The reliable trick: put 64 zeros in the formula's
   `sha256`, run `brew fetch`, and read the "checksum of downloaded file" it
   reports:
   ```sh
   brew fetch tomoima525/sotto/sotto   # prints the real sha256 on mismatch
   ```

## Generating the dependency resources

`brew update-python-resources` **does not work here** — it pins resolution with
`--uploaded-prior-to` and fails to resolve this tree. Generate the resources
from a plain pip dry-run report instead:

```sh
python3.12 -m pip install --dry-run --ignore-installed \
  --report /tmp/pipreport.json \
  "https://github.com/tomoima525/sotto/archive/refs/tags/1.0.2.tar.gz"
```

Then emit one `resource` block per entry in `/tmp/pipreport.json`
(`download_info.url` + `archive_info.hashes.sha256`), with these rules:

- **Exclude the torch subtree**: `torch`, `sympy`, `networkx`, `mpmath`.
  `mlx-whisper` declares `torch` but only imports it in `torch_whisper.py` (the
  unused PyTorch reference impl); the transcribe path never touches it.
  Excluding it removes ~600 MB. (`scipy` and `numba`/`llvmlite` **are** used —
  keep them.)
- **Exclude `sotto` itself** (it's the main package, not a resource).
- **Hyphenate resource names** to the PyPI-canonical form
  (`huggingface_hub` → `huggingface-hub`, `tomli_w` → `tomli-w`,
  `typing_extensions` → `typing-extensions`), or `brew audit` rejects them.

Result: ~55 wheel resources (plus the `rumps` sdist).

## Install method: pip-install cached downloads (not resource staging)

The standard `virtualenv_install_with_resources` **fails** on this tree:
Homebrew's resource staging unzips binary wheels (`cp312-*-arm64.whl`) into a
directory with no `setup.py`/`pyproject.toml`, so `pip install <dir>` errors
with *"is not installable"*. `using: :nounzip` does **not** prevent this, and
`r.cached_download` alone fails too — its cache filename drops the `.whl`
extension, so pip rejects it as an *"invalid wheel filename"*.

The working approach copies each cached download to its real filename and
pip-installs the file, sidestepping staging entirely:

```ruby
def install
  venv = virtualenv_create(libexec, "python3.12")
  resources.each do |r|
    r.fetch
    wheel = buildpath/File.basename(r.url)
    cp r.cached_download, wheel
    venv.pip_install wheel
  end
  venv.pip_install_and_link buildpath
end
```

`venv.pip_install` passes `--no-deps`, so the resource set must be the complete
closure (it is — only the unused torch subtree is omitted).

## Creating / updating the tap

```sh
# one time
gh repo create tomoima525/homebrew-sotto --public --description "Homebrew tap for Sotto"
brew tap tomoima525/sotto
```

Put the assembled formula at `Formula/sotto.rb` in the tap, then commit + push.

## Testing the formula locally

```sh
brew install --build-from-source tomoima525/sotto/sotto
sotto --help          # also part of the formula's `test do` block
sotto --version       # asserts the formula version
sotto download        # one-time model fetch (~4 GB)
sotto run             # menu bar app
brew audit --tap=tomoima525/sotto --formula sotto
brew test tomoima525/sotto/sotto
```

## Permissions

A formula install runs `sotto` as a CLI, so macOS attributes TCC permissions to
the **terminal** that launches it (Microphone, Input Monitoring, Accessibility) —
same as development. Document this for users, or steer them to the future Cask
for a self-contained app that prompts for its own permissions.

## Updating for new releases

If dependencies are **unchanged**, only `url` + `sha256` change — re-run
`brew fetch` for the new sha and bump those two lines. If dependencies
changed, regenerate the resources via the pip-report method above.
