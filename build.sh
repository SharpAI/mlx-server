#!/bin/bash
set -eo pipefail

echo "=============================================="
echo "    SwiftLM Build Script                      "
echo "=============================================="

# --- 1. Submodules ---
echo ""
echo "=> [1/4] Initializing submodules..."
git submodule update --init --recursive

# --- 2. Check for cmake and resolve Swift dependencies ---
echo ""
echo "=> [2/4] Checking dependencies and resolving packages..."
swift package resolve
echo "=> [2/4] Checking build dependencies..."
if ! command -v cmake &> /dev/null; then
    echo "cmake not found. Installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is required to install cmake."
        echo "   Install Homebrew: https://brew.sh"
        exit 1
    fi
    brew install cmake
fi
echo "   cmake: $(cmake --version | head -1)"

# --- 3. Build the Metal kernel library (mlx.metallib) from source ---
echo ""
echo "=> [3/4] Building Metal kernels (mlx.metallib)..."

MLX_SRC=".build/checkouts/mlx-swift/Source/Cmlx/mlx"
METALLIB_BUILD_DIR=".build/metallib_build"
METALLIB_DEST=".build/arm64-apple-macosx/release"

rm -rf "$METALLIB_BUILD_DIR"
mkdir -p "$METALLIB_BUILD_DIR"

pushd "$METALLIB_BUILD_DIR" > /dev/null

cmake "../../$MLX_SRC" \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DMLX_METAL_JIT=OFF \
    -DMLX_ENABLE_NAX=1 \
    -DCMAKE_BUILD_TYPE=Release \
    2>&1 | tail -40

echo "   Compiling Metal shaders..."
if ! make mlx-metallib -j$(sysctl -n hw.ncpu) 2>&1 | tail -80; then
    echo "❌ Failed to build mlx.metallib. If you see 'missing Metal Toolchain', run:"
    echo "   xcodebuild -downloadComponent MetalToolchain"
    exit 1
fi

popd > /dev/null

# Copy the freshly built metallib next to the binary
mkdir -p "$METALLIB_DEST"
if [ -f "$METALLIB_BUILD_DIR/lib/mlx.metallib" ]; then
    cp "$METALLIB_BUILD_DIR/lib/mlx.metallib" "$METALLIB_DEST/mlx.metallib"
    echo "✅ Built and copied mlx.metallib to $METALLIB_DEST/"
elif [ -f "$METALLIB_BUILD_DIR/mlx.metallib" ]; then
    cp "$METALLIB_BUILD_DIR/mlx.metallib" "$METALLIB_DEST/mlx.metallib"
    echo "✅ Built and copied mlx.metallib to $METALLIB_DEST/"
else
    # Search for it anywhere in the build dir
    BUILT=$(find "$METALLIB_BUILD_DIR" -name "mlx.metallib" | head -1)
    if [ -n "$BUILT" ]; then
        cp "$BUILT" "$METALLIB_DEST/mlx.metallib"
        echo "✅ Built and copied mlx.metallib to $METALLIB_DEST/"
    else
        echo "❌ Failed to build mlx.metallib. Check cmake output above."
        exit 1
    fi
fi

# --- 4. Build SwiftLM ---
echo ""
echo "=> [4/4] Building SwiftLM (release)..."
swift build -c release

echo ""
echo "=============================================="
echo "✅ Build complete!"
echo "   Binary:   .build/release/SwiftLM"
echo "   Metallib: $METALLIB_DEST/mlx.metallib"
echo "=============================================="
