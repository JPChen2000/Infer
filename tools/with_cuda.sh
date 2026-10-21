#!/usr/bin/env bash
# Run with user-space driver libraries matching the currently loaded module.
# --prepare downloads and extracts NVIDIA's matching driver; no system install.
set -euo pipefail
prepare=false
if [[ ${1:-} == --prepare ]]; then prepare=true; shift; fi
if [[ ${1:-} == -- ]]; then shift; fi
if (( $# == 0 )); then
    echo "Usage: $0 [--prepare] [--] command [args ...]" >&2
    exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    exec "$@"
fi
version=$(cat /sys/module/nvidia/version 2>/dev/null || true)
if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "No loaded NVIDIA module; install/load the host driver first." >&2
    exit 1
fi
driver_dir="${XDG_CACHE_HOME:-$HOME/.cache}/infer/nvidia/$version/lib"
if [[ ! -f $driver_dir/libcuda.so.1 || ! -f $driver_dir/libnvidia-ml.so.1 ]]; then
    if ! $prepare; then
        echo "Driver/library mismatch. Run $0 --prepare -- command to cache NVIDIA $version libraries." >&2
        exit 1
    fi
    task_dir=$(mktemp -d)
    trap 'rm -rf "$task_dir"' EXIT
    curl --proto '=https' -fL --retry 2 -o "$task_dir/driver.run" \
        "https://download.nvidia.com/XFree86/Linux-x86_64/$version/NVIDIA-Linux-x86_64-$version.run"
    sh "$task_dir/driver.run" --extract-only --target "$task_dir/extracted" >/dev/null
    mkdir -p "$driver_dir"
    for library in libcuda libnvidia-ml libnvidia-ptxjitcompiler libnvidia-nvvm libnvidia-gpucomp; do
        cp "$task_dir/extracted/$library.so.$version" "$driver_dir/"
        ln -sfn "$library.so.$version" "$driver_dir/$library.so.1"
        ln -sfn "$library.so.$version" "$driver_dir/$library.so"
    done
    rm -rf "$task_dir"
    trap - EXIT
fi
export LD_LIBRARY_PATH="$driver_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv >&2
exec "$@"
