class Pvx < Formula
  include Language::Python::Virtualenv

  desc "Phase-vocoder DSP toolkit with pvx command suite"
  homepage "https://github.com/TheColby/pvx"
  license "MIT"

  stable_url = "__PVX_STABLE_URL__"
  stable_sha = "__PVX_STABLE_SHA__"
  stable_version = "__PVX_STABLE_VERSION__"
  if stable_url != "__PVX_STABLE_URL__"
    url stable_url
    sha256 stable_sha
    version stable_version
  end

  head "https://github.com/TheColby/pvx.git", branch: "main"

  depends_on "python@3.12"
  depends_on "libsndfile"
  uses_from_macos "libffi"

  def install
    venv = virtualenv_create(libexec, "python3.12")

    # This tap formula intentionally resolves Python runtime dependencies from
    # PyPI at install time. It is suitable for this project's tap/raw formula
    # flow; generating fully vendored resources is handled separately from the
    # main release path.
    venv.pip_install %w[
      numpy>=1.24
      soundfile>=0.12.1
      scipy>=1.10
      librosa>=0.10.2
      pyloudnorm>=0.1.1
    ]
    venv.pip_install_and_link buildpath

    man1.install Dir[buildpath/"man/man1/*.1"] if (buildpath/"man/man1").exist?
  end

  test do
    assert_match "Unified entry point", shell_output("#{bin}/pvx --help")
    assert_match "phase vocoder", shell_output("#{bin}/pvxvoc --help")
  end
end
