#!/bin/sh
# Builds libgdeflate (GDeflate tile decompression, for Portal RTX / RTX IO packages) next to
# this script. Fetches NVIDIA's libdeflate fork (Apache-2.0 / MIT) at the commit Microsoft's
# DirectStorage GDeflate uses. Needs git and a C compiler.
#   Linux/macOS:  sh build.sh
#   Windows:      run from Git Bash / MSYS2 with gcc, produces gdeflate.dll
set -e
cd "$(dirname "$0")"
if [ ! -d libdeflate ]; then
  git clone -q https://github.com/NVIDIA/libdeflate libdeflate
  git -C libdeflate checkout -q 8ba9502fb30d2bf728592d121f0d402e40c8cb05
fi
L=libdeflate
SRC="$L/lib/gdeflate_decompress.c $L/lib/utils.c $L/lib/x86/cpu_features.c gdeflate_tiles.c"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) OUT=gdeflate.dll ;;
  Darwin) OUT=libgdeflate.dylib ;;
  *) OUT=libgdeflate.so ;;
esac
${CC:-cc} -O2 -shared -fPIC -I$L $SRC -o $OUT
echo "built $(pwd)/$OUT"
