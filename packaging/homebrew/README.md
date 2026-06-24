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

## Releasing a version the formula can point at

The formula's `url` is a tagged GitHub source tarball, so each formula update
needs a matching tag/release.

1. Bump `version` in `pyproject.toml` (already `1.0.2`). `sotto.__version__` is
   derived from the installed package metadata, so it follows automatically.
2. Merge to `main`, then tag and release:
   ```sh
   git tag 1.0.2 && git push origin 1.0.2
   gh release create 1.0.2 --generate-notes
   ```
3. Get the tarball sha256 for the formula:
   ```sh
   brew fetch --build-from-source ./packaging/homebrew/sotto.rb   # prints sha256
   # or: curl -sL https://github.com/tomoima525/sotto/archive/refs/tags/1.0.2.tar.gz | shasum -a 256
   ```

## Creating / updating the tap

1. Create the tap repo once:
   ```sh
   gh repo create tomoima525/homebrew-sotto --public \
     --description "Homebrew tap for Sotto"
   ```
2. Copy `sotto.rb` into the tap as `Formula/sotto.rb` and set `url` + `sha256`.
3. Generate the dependency `resource` blocks (this is the bulk of the formula):
   ```sh
   brew tap tomoima525/sotto
   brew update-python-resources sotto
   ```
   This resolves every PyPI dependency (mlx, mlx-whisper, mlx-lm, transformers,
   tokenizers, safetensors, torch, numpy, scipy, huggingface-hub, rumps,
   sounddevice, the pyobjc-* frameworks, and all transitive deps) and writes
   `resource` stanzas. Many are wheels — expected on arm64; some (mlx, torch,
   tokenizers) are wheel-only. Re-run this command on every version bump.

## Testing the formula locally

```sh
brew install --build-from-source tomoima525/sotto/sotto
sotto --help          # also runs as the formula's `test do` block
sotto download        # one-time model fetch (~4 GB)
sotto run             # menu bar app
brew audit --strict --online tomoima525/sotto/sotto
```

## Permissions

A formula install runs `sotto` as a CLI, so macOS attributes TCC permissions to
the **terminal** that launches it (Microphone, Input Monitoring, Accessibility) —
same as development. Document this for users, or steer them to the future Cask
for a self-contained app that prompts for its own permissions.

## Updating for new releases

Bump `pyproject.toml`, tag + release, then in the tap update `url`, `sha256`, and
re-run `brew update-python-resources sotto`. `brew bump-formula-pr` can automate
the url/sha bump.
