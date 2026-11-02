"""Block-level weight streaming: keep only one transformer block's weights in memory at a time.

On Apple Silicon the CPU and GPU share memory, so "offloading" means reading weights back
from the SSD. Each streamed block is a stand-in that, when called, takes its block's tensors
(read ahead on a background thread while the previous block ran), runs the block, forces the
result, and drops the weights again. The reads go through the file cache, which macOS can
reclaim at any time without swapping other apps out, so this mode trades a little speed for
a much lower peak.
"""

from __future__ import annotations

import json
import struct
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

_HOST_DTYPES = {
    "BF16": np.uint16,  # viewed as bfloat16 once in MLX
    "F16": np.float16,
    "F32": np.float32,
    "U32": np.uint32,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
}


class TensorIndex:
    """Offsets of every tensor in a component's safetensors shards, read from the headers only."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._entries: dict[str, tuple[Path, int, str, tuple[int, ...], int]] = {}
        self.quantization: int | None = None
        for shard in self._shards():
            with shard.open("rb") as fh:
                header_len = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(header_len))
            meta = header.pop("__metadata__", None) or {}
            if self.quantization is None and meta.get("quantization_level") not in (None, "None", "null", ""):
                self.quantization = int(meta["quantization_level"])
            for name, info in header.items():
                start, end = info["data_offsets"]
                self._entries[name] = (shard, 8 + header_len + start, info["dtype"], tuple(info["shape"]), end - start)

    def _shards(self) -> list[Path]:
        index = self.directory / "model.safetensors.index.json"
        if index.exists():
            names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            return [self.directory / n for n in names]
        return sorted(p for p in self.directory.glob("*.safetensors") if not p.name.startswith("._"))

    def keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._entries if k.startswith(prefix)]

    def nbytes(self, prefix: str = "") -> int:
        return sum(e[4] for k, e in self._entries.items() if k.startswith(prefix))

    def read_host(self, keys: list[str], strip: str = "") -> dict[str, tuple[np.ndarray, str]]:
        """Plain file reads into host buffers; touches no MLX state, so it is safe on a worker thread.

        Not mmap: faulting in thousands of mapped pages per block contends on the VM system and
        made other apps' timers late while streaming (bench probe).
        """
        out = {}
        handles: dict[Path, object] = {}
        try:
            for key in keys:
                path, offset, dtype, shape, _ = self._entries[key]
                buffer = np.empty(shape or (), dtype=_HOST_DTYPES[dtype])
                fh = handles.get(path) or handles.setdefault(path, path.open("rb", buffering=0))
                fh.seek(offset)
                fh.readinto(memoryview(buffer).cast("B"))
                out[key[len(strip):]] = (buffer, dtype)
        finally:
            for fh in handles.values():
                fh.close()
        return out

    def read_copied(self, keys: list[str], strip: str = "") -> dict[str, tuple[mx.array, str]]:
        """read_host, then copied into MLX buffers. Builds no graph, so it also runs on the worker."""
        return {name: (mx.array(buffer), dtype) for name, (buffer, dtype) in self.read_host(keys, strip).items()}

    def read(self, keys: list[str], strip: str = "") -> dict[str, mx.array]:
        return as_weights(self.read_copied(keys, strip))


def as_weights(copied: dict[str, tuple[mx.array, str]]) -> dict[str, mx.array]:
    return {name: mx.view(array, mx.bfloat16) if dtype == "BF16" else array for name, (array, dtype) in copied.items()}


def block_prefixes(index: TensorIndex, lists: list[str]) -> list[str]:
    """Weight prefixes for every listed block stack, including nested module paths."""
    found = set()
    for name in lists:
        prefix = f"{name}."
        for key in index.keys(prefix):
            index_part = key[len(prefix):].split(".", 1)[0]
            if index_part.isdigit():
                found.add((name, int(index_part)))
    return [f"{name}.{i}." for name, i in sorted(found, key=lambda f: (lists.index(f[0]), f[1]))]


def stacks_from_index(index: TensorIndex, min_blocks: int = 4) -> list[str]:
    """Block stacks named by the weights themselves: the path before the first number, if it has enough blocks."""
    numbers: dict[str, set[int]] = {}
    for key in index.keys():
        parts = key.split(".")
        for i, part in enumerate(parts[:-1]):
            if part.isdigit():
                numbers.setdefault(".".join(parts[:i]), set()).add(int(part))
                break
    return sorted(path for path, found in numbers.items() if path and len(found) >= min_blocks)


def stream_peak_bytes(directory: Path, lists: list[str]) -> int:
    """Weights resident while streaming: the non-block part, the running block and the one read ahead."""
    index = TensorIndex(directory)
    if lists == ["auto"]:
        lists = stacks_from_index(index)
    prefixes = block_prefixes(index, lists)
    blocks = [index.nbytes(p) for p in prefixes]
    return index.nbytes() - sum(blocks) + 2 * max(blocks, default=0)


def _quantize_like(module: nn.Module, weights: dict, bits: int | None) -> None:
    """Give `module` the same (possibly mixed) quantized structure the stored weights have."""
    from mflux.models.common.weights.loading.weight_applier import WeightApplier

    if bits is None:
        return
    nn.quantize(module, class_predicate=WeightApplier._stored_layer_predicate(weights, None), bits=bits)


def _evaluated(value):
    arrays = [v for _, v in tree_flatten(value) if isinstance(v, mx.array)]
    if arrays:
        mx.eval(arrays)


class _ReadAhead:
    """Reads the next block's weights on a worker thread while the current block runs on the GPU.

    Without it the GPU idles during every read; the clock drops and the next block's long
    kernels then hold up other apps' GPU work (bench probe GPU p99 ~30-90 ms on edits vs ~3 ms).
    """

    def __init__(self, index: TensorIndex):
        self._index = index
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dmlx-read-ahead")
        self._pending: tuple[str, Future] | None = None

    def take(self, prefix: str, keys: list[str]) -> dict:
        pending, self._pending = self._pending, None
        if pending is not None and pending[0] == prefix:
            return pending[1].result()
        return self._index.read_copied(keys, prefix)

    def start(self, prefix: str, keys: list[str]) -> None:
        self._pending = (prefix, self._worker.submit(self._index.read_copied, keys, prefix))


class StreamedBlock(nn.Module):
    """Stand-in for one block: take its weights, run it, force the result, drop the weights."""

    def __init__(
        self, block: nn.Module, index: TensorIndex, prefix: str, bits: int | None,
        reader: _ReadAhead, check=None,
    ):
        super().__init__()
        # Kept outside the module tree so the parent's parameters() never sees an unloaded block.
        object.__setattr__(self, "_block", block)
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_bits", bits)
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_check", check)
        object.__setattr__(self, "_skeleton", None)
        object.__setattr__(self, "_keys", index.keys(prefix))
        object.__setattr__(self, "_next", None)  # (prefix, keys) of the block that runs after this one

    def __getattr__(self, key):
        """Expose block metadata used by model-level inference context managers."""
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_block"), key)

    def __call__(self, *args, **kwargs):
        copied = self._reader.take(self._prefix, self._keys)
        if self._next is not None:
            self._reader.start(*self._next)
        weights = tree_unflatten(list(as_weights(copied).items()))
        del copied
        if self._skeleton is None:
            _quantize_like(self._block, weights, self._bits)
            # Unevaluated placeholders: swapping them back in releases the real weights.
            object.__setattr__(self, "_skeleton", tree_map(lambda x: x, self._block.parameters()))
        self._block.update(weights, strict=False)
        del weights
        try:
            out = self._block(*args, **kwargs)
            _evaluated(out)
            if self._check is not None:
                self._check()
            cache = kwargs.get("kv_cache")
            if cache is not None:
                # The reference K/V a block stores during extraction would otherwise keep its weights alive.
                _evaluated([getattr(cache, "_double", []), getattr(cache, "_single", [])])
        finally:
            self._block.update(self._skeleton, strict=False)
        return out


class SkippedLayer(nn.Module):
    """Stand-in for a decoder layer whose output is never used (FLUX.2 reads Qwen3 layers 9, 18, 27 only)."""

    def __call__(self, hidden_states, *args, **kwargs):
        return hidden_states, None


def find_stacks(module: nn.Module, prefix: str = "", min_blocks: int = 4) -> list[str]:
    """Dotted paths of the block stacks in a module (lists of at least `min_blocks` sub-modules)."""
    found = []
    for name, child in module.children().items():
        if isinstance(child, list) and len(child) >= min_blocks and all(isinstance(c, nn.Module) for c in child):
            found.append(prefix + name)
        elif isinstance(child, nn.Module):
            found += find_stacks(child, f"{prefix}{name}.", min_blocks)
    return found


def make_streamed(
    module: nn.Module, directory: Path, lists: list[str], skip_from: dict[str, int] | None = None, check=None,
) -> int:
    """Load `module`'s non-block weights and turn its block stacks into streamed stand-ins.

    Returns the bytes now resident (the non-block weights). `skip_from` maps a stack name to the
    first index whose output is unused; those layers are never read at all.
    """
    index = TensorIndex(directory)
    prefixes = block_prefixes(index, lists)
    in_blocks = set()
    for p in prefixes:
        in_blocks.update(index.keys(p))
    resident_keys = [k for k in index.keys() if k not in in_blocks]
    resident = tree_unflatten(list(index.read(resident_keys).items()))
    bits = index.quantization
    # Quantize only the non-block part here; each block gets its structure on first use.
    def parent_and_attribute(path: str):
        parent = module
        parts = path.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    locations = {name: parent_and_attribute(name) for name in lists}
    blocks = {name: getattr(parent, attribute) for name, (parent, attribute) in locations.items()}
    for name in lists:
        parent, attribute = locations[name]
        setattr(parent, attribute, [])
    _quantize_like(module, resident, bits)
    module.update(resident, strict=False)
    mx.eval(module.parameters())
    resident_bytes = sum(v.nbytes for _, v in tree_flatten(module.parameters()))
    reader = _ReadAhead(index)
    streamed: list[StreamedBlock] = []
    for name, stack in blocks.items():
        stop = (skip_from or {}).get(name, len(stack))
        replaced = []
        for i, block in enumerate(stack):
            if i < stop:
                replaced.append(StreamedBlock(block, index, f"{name}.{i}.", bits, reader, check=check))
                streamed.append(replaced[-1])
            else:
                replaced.append(SkippedLayer())
        parent, attribute = locations[name]
        setattr(parent, attribute, replaced)
    # Blocks run in stack order, and the next forward pass starts again from the first block.
    for current, following in zip(streamed, streamed[1:] + streamed[:1]):
        object.__setattr__(current, "_next", (following._prefix, following._keys))
    return resident_bytes
