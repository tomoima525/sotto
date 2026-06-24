# Skeleton Homebrew formula for Sotto. The real, resource-filled copy lives in
# the tap repo (tomoima525/homebrew-sotto) as Formula/sotto.rb. This copy is
# kept in-repo for review and reproducibility; see ./README.md for how the
# `resource` blocks are generated and the install method below is derived.
#
# Apple Silicon only (MLX has no Intel build). sounddevice bundles its own
# PortAudio, so no system audio dependency is required. The ~4 GB models are
# not bundled; they download on first run (`sotto download`).
class Sotto < Formula
  include Language::Python::Virtualenv

  desc "Fully-local dictation for macOS: hotkey to speak, get cleaned text"
  homepage "https://github.com/tomoima525/sotto"
  url "https://github.com/tomoima525/sotto/archive/refs/tags/1.0.2.tar.gz"
  sha256 "7d2cb50de567782b8de1d4a05b35ae1f78dec36502bb543fc00865cd1a2fea21"
  license "MIT"

  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.12"

  # >>> ~55 resource blocks go here (generated; see ./README.md) <<<
  # The full dependency closure as wheels, EXCEPT the torch subtree
  # (torch/sympy/networkx/mpmath), which mlx-whisper declares but never imports
  # on the transcribe path — excluding it removes ~600 MB.
  #
  #   resource "cffi" do
  #     url "https://files.pythonhosted.org/packages/.../cffi-2.0.0-cp312-cp312-macosx_11_0_arm64.whl"
  #     sha256 "..."
  #   end
  #   ... (mlx, mlx-whisper, mlx-lm, numba, scipy, transformers, tokenizers,
  #        safetensors, numpy, huggingface-hub, rumps, sounddevice, pyobjc-*, ...)

  def install
    venv = virtualenv_create(libexec, "python3.12")
    # Install each resource from its downloaded file rather than via brew's
    # resource staging: staging unzips binary wheels (cp312-*-arm64.whl) into a
    # directory with no setup.py/pyproject.toml, which breaks `pip install`.
    # (`using: :nounzip` does not prevent this.) Copying the cached download to
    # its real filename and pip-installing the file sidesteps staging entirely.
    resources.each do |r|
      r.fetch
      wheel = buildpath/File.basename(r.url)
      cp r.cached_download, wheel
      venv.pip_install wheel
    end
    venv.pip_install_and_link buildpath
  end

  test do
    # Exercises the CLI without needing a mic, models, or TCC permissions.
    assert_match "usage: sotto", shell_output("#{bin}/sotto --help")
    assert_match version.to_s, shell_output("#{bin}/sotto --version")
  end
end
