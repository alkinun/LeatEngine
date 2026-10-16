#!/bin/sh
# Installs LeatEngine on Linux x86-64 from a release: its self-contained tarball, checked against
# the release's SHA256SUMS, unpacked to ~/.local/share/leatengine/VERSION and linked as
# ~/.local/bin/leatengine. An upgrade keeps the version installed until the new one runs.
#
#   curl -fsSL https://github.com/alkinun/LeatEngine/releases/latest/download/install.sh | sh
#   sh install.sh [--version VERSION | --from URL] [--uninstall]
#
# --version installs a release other than the latest; --from, the tarball in a directory of that
# URL beside its SHA256SUMS, as file:///path/to/dist of scripts/build_release.py. XDG_DATA_HOME,
# where set, takes the place of ~/.local/share. Piped to sh, the options follow `sh -s --`.
set -eu

releases=https://github.com/alkinun/LeatEngine/releases
from=$releases/latest/download
uninstall=
while [ $# -gt 0 ]; do
    case $1 in
        --version) from=$releases/download/v${2:?--version takes a version, as 0.1.0}; shift 2 ;;
        --from) from=${2:?--from takes a URL}; shift 2 ;;
        --uninstall) uninstall=1; shift ;;
        *) printf 'install.sh: unknown option %s\n' "$1" >&2; exit 2 ;;
    esac
done

data=${XDG_DATA_HOME:-$HOME/.local/share}/leatengine
bin=$HOME/.local/bin
link=$bin/leatengine

die() {
    printf 'install.sh: %s\n' "$*" >&2
    exit 1
}

# whether the command is this installer's, a link into its versions
ours() {
    case $(readlink "$link" 2>/dev/null) in "$data"/*) return 0 ;; esac
    return 1
}

if [ -n "$uninstall" ]; then
    if ours; then rm -f "$link"; fi
    rm -rf "$data"
    echo "Uninstalled LeatEngine. Models pulled, in ~/.cache/huggingface, and the kernels it" \
        "compiled, in ~/.cache/tinygrad, are kept."
    exit 0
fi

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] ||
    die "the release is for Linux on x86-64, not $(uname -s) on $(uname -m)"
if [ -e "$link" ] || [ -L "$link" ]; then
    ours || die "$link is another install's, as uv tool's: uninstall that first"
fi
for tool in tar sha256sum mktemp; do
    command -v "$tool" >/dev/null 2>&1 || die "needs $tool"
done
if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fsSL --retry 3 "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q "$1" -O "$2"; }
else
    die "needs curl or wget"
fi

# everything is staged beside the versions, and gone on any exit
mkdir -p "$data" "$bin"
stage=$(mktemp -d "$data/.install.XXXXXX")
trap 'rm -rf "$stage"' EXIT
trap 'exit 130' INT TERM

fetch "$from/SHA256SUMS" "$stage/SHA256SUMS" || die "found no SHA256SUMS at $from"
tarball=$(sed -n 's/^[0-9a-f]\{64\}  \(leatengine-[^ /]*-linux-x86_64\.tar\.gz\)$/\1/p' \
    "$stage/SHA256SUMS" | head -n 1)
[ -n "$tarball" ] || die "$from/SHA256SUMS names no tarball for Linux x86-64"
version=${tarball#leatengine-}
version=${version%-linux-x86_64.tar.gz}
echo "Downloading LeatEngine $version"
fetch "$from/$tarball" "$stage/$tarball" || die "could not download $from/$tarball"
(cd "$stage" && grep -F "  $tarball" SHA256SUMS | sha256sum --check --quiet -) ||
    die "$tarball does not match its SHA-256 in SHA256SUMS"
tar -xzf "$stage/$tarball" -C "$stage"
new=$stage/leatengine-$version-linux-x86_64
[ -x "$new/bin/leatengine" ] || die "$tarball holds no bin/leatengine"

# the new version replaces the old only once it runs here: doctor reports, whatever it finds of
# the machine, and exits 1 where it finds no GPU ready
report=$("$new/bin/leatengine" doctor --json 2>/dev/null) || true
case $report in
    *"\"leatengine\": \"$version\""*) ;;
    *) die "LeatEngine $version does not run here: its doctor gave no report; kept what was" \
        "installed" ;;
esac

dest=$data/$version
if [ -e "$dest" ]; then mv "$dest" "$stage/installed"; fi # the same version, again
mv "$new" "$dest"
ln -sfn "$dest/bin/leatengine" "$link"
for other in "$data"/*; do
    if [ "$other" != "$dest" ]; then rm -rf "$other"; fi
done
echo "Installed LeatEngine $version in $dest, run as $link."
case :$PATH: in
    *":$bin:"*) ;;
    *) echo "$bin is not on your PATH: add it in your shell's profile." ;;
esac
echo "Next: leatengine doctor"
