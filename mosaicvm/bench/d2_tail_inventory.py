"""Read actual local shard headers; no model loading, downloads, or GPU work."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import struct


def inventory(snapshot: Path) -> dict:
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    missing = sorted(s for s in set(index.values()) if not (snapshot / s).is_file())
    if missing:
        raise FileNotFoundError(f'Missing shard targets: {missing}')
    groups = Counter()
    tensors = Counter()
    routed_weights = routed_scales = head_fp32 = 0
    total = 0
    for shard in sorted(set(index.values())):
        with (snapshot / shard).open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f'Truncated header: {shard}')
            length = struct.unpack('<Q', raw)[0]
            if length > (snapshot / shard).stat().st_size - 8:
                raise ValueError(f'Invalid header length: {shard}')
            header = json.loads(f.read(length))
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            start, end = tensor['data_offsets']
            if not 0 <= start <= end <= (snapshot / shard).stat().st_size - 8 - length:
                raise ValueError(f'Invalid tensor offsets: {name}')
            size = end - start
            total += size
            if name == 'head.weight':
                head_fp32 = 4
                for dim in tensor['shape']:
                    head_fp32 *= dim
            if not name.startswith(('layers.41.', 'layers.42.')):
                continue
            group = ('routed' if '.ffn.experts.' in name else
                     'shared' if '.shared_experts.' in name else 'core_router_norm')
            groups[group] += size
            tensors[group] += 1
            if group == 'routed':
                if name.endswith('.weight'):
                    routed_weights += size
                elif name.endswith('.scale'):
                    routed_scales += size
                else:
                    raise ValueError(f'Unaccounted routed tensor: {name}')
    # This checkpoint has 2 layers × 256 experts × 3 linears × weight/scale.
    assert tensors['routed'] == 3072, tensors
    assert routed_weights == routed_scales * 16, (routed_weights, routed_scales)
    return {
        'classification': 'LOCAL_SOURCE_STORAGE_INVENTORY_NOT_RUNTIME_TIMING',
        'snapshot': str(snapshot.resolve()),
        'shards_read': len(set(index.values())),
        'checkpoint_payload_bytes': total,
        'tail_source_payload_bytes': dict(groups),
        'tail_tensor_counts': dict(tensors),
        'tail_source_total_bytes': sum(groups.values()),
        'derived_representation_bytes': {
            'routed_packed_weights': routed_weights,
            'routed_e8m0_scales': routed_scales,
            'routed_vnni_v4b_int8_weights_fp32_scales': 2 * routed_weights + 4 * routed_scales,
            'fp32_head': head_fp32,
        },
        'excluded': ['runtime dtype expansion', 'KV/DSA', 'scratch', 'allocator/runtime',
                     'student', 'verifier residency', 'duplicate canonical/expanded buffers'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path.home() / (
        '.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/'
        'snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/d2_tail_inventory.json'))
    args = parser.parse_args()
    result = inventory(args.snapshot)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
