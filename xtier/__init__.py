"""XTier: cross-tier CPU/GPU hybrid MoE inference for SGLang."""

__version__ = "0.1.1"

# The SGLang main commit this XTier is tested with (2026-10-30): README, tools/check_sglang.py, a warning at load
SGLANG_COMMIT = "6162b07199"

# The SGLang versions that carry no commit hash and are still known to XTier: the releases the Turing tier was
# measured on ("0.5.22") and the sm_75 build of a release ("0.5.10+sm75", the fork whose sgl-kernel is compiled for
# compute_75 only). The hash regex (+g<hash>) finds nothing in them, so the checks would call every such SGLang
# "unknown" and fail --strict. The sm_75 marker matters here: that build's kernels exist only for sm_75 (no FA3, no
# sm_90 code), so it is the right SGLang for a Turing host, and XTier's Turing tier (docs/sm75.md) goes with it.
SGLANG_KNOWN_VERSIONS = ('0.5.10+sm75', '0.5.22')


def sglang_version_kind(version):
    """How the installed SGLang compares with the one XTier is tested with: 'commit' (the tested main commit),
    'known' (a version in SGLANG_KNOWN_VERSIONS: a release or the sm_75 build), 'other' (anything else)."""
    import re

    m = re.search(r'\+g([0-9a-f]{7,})', version)
    if m:
        return 'commit' if m.group(1)[:7] == SGLANG_COMMIT[:7] else 'other'
    return 'known' if version in SGLANG_KNOWN_VERSIONS else 'other'
