#!/bin/bash
# audit-uv-cache.sh

function naive_first(){
  CACHE_DIR=${1:-~/.cache/uv}

  echo "=== Taille totale ==="
  du -sh "$CACHE_DIR"

  echo ""
  echo "=== Versions Python détectées ==="
  find "$CACHE_DIR/wheels" -name "*.whl" 2>/dev/null | \
    grep -oP 'cp3\d+' | \
    sort -u | \
    sed 's/cp3\([0-9]*\)/Python 3.\1/'

  echo ""
  echo "=== Packages présents ==="
  find "$CACHE_DIR/wheels" -name "*.whl" 2>/dev/null | \
    grep -oP '^.*/\K[a-zA-Z0-9_]+-[0-9]+\.[0-9]+' | \
    sort -u

  echo ""
  echo "=== Taille par version Python ==="
  for tag in cp39 cp310 cp311 cp312; do
    size=$(find "$CACHE_DIR/wheels" -name "*${tag}*" | \
      xargs du -sc 2>/dev/null | tail -1 | cut -f1)
    echo "  Python ${tag}: ${size:-0} KB"
  done
}


CACHE_DIR=${1:-~/.cache/uv}
ARCHIVE="$CACHE_DIR/archive-v0"

echo "=== Taille totale du cache ==="
du -sh "$CACHE_DIR"
echo ""
du -sh "$CACHE_DIR"/*/  | sort -h
echo ""

echo "=== Packages et versions Python compatibles ==="
find "$ARCHIVE" -name "WHEEL" | while read wheel_file; do
    # Extraire le nom du package depuis le dossier dist-info
    dist_info=$(dirname "$wheel_file")
    package=$(basename "$dist_info" | sed 's/-[0-9].*//')
    version=$(basename "$dist_info" | grep -oP '[0-9]+\.[0-9]+[^\.]*(\..*)?(?=\.dist-info)')

    # Extraire les tags Python
    tags=$(grep "^Tag:" "$wheel_file" | grep -oP 'cp3\d+|py3|none' | sort -u | tr '\n' ' ')

    echo "$package==$version | python: ${tags:-pure-python}"
done | sort -u
echo ""

echo "=== Versions Python détectées dans le cache ==="
find "$ARCHIVE" -name "WHEEL" | xargs grep -h "^Tag:" 2>/dev/null | \
    grep -oP 'cp3\d+' | sort -u | \
    sed 's/cp3\([0-9]*\)/Python 3.\1/'
echo ""

echo "=== Top 10 packages les plus lourds ==="
find "$ARCHIVE" -name "*.dist-info" -type d | while read dist_info; do
    package=$(basename "$dist_info" | sed 's/\.dist-info//')
    parent=$(dirname "$dist_info")
    size=$(du -sk "$parent" | cut -f1)
    echo "$size $package"
done | sort -nr | head -10 | awk '{printf "%-20s %s\n", $1" K", $2 }'
