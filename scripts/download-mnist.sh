#!/bin/bash
set -e

MNIST_URL="https://ossci-datasets.s3.amazonaws.com/mnist"
DATA_DIR="data/mnist"
mkdir -p "$DATA_DIR"

FILES=(
  "train-images-idx3-ubyte"
  "train-labels-idx1-ubyte"
  "t10k-images-idx3-ubyte"
  "t10k-labels-idx1-ubyte"
)

for file in "${FILES[@]}"; do
  if [ -f "$DATA_DIR/$file" ]; then
    echo "$file already exists, skipping."
  else
    echo "Downloading $file..."
    curl -sSL "$MNIST_URL/$file.gz" | gunzip > "$DATA_DIR/$file"
    echo "  -> $(wc -c < "$DATA_DIR/$file") bytes"
  fi
done

echo "Done. MNIST data is in $DATA_DIR/"
