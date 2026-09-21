"""Full CPU tail adapter comparison on captured states; not native certification."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from mosaicvm.bench.source_tail import SourceTail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    c = torch.load(a.capture, map_location='cpu', weights_only=True)
    tail = SourceTail(device='cpu')
    if (c.get('teacher_class') != 'SOURCE_CLASS_TORCH_SHIM'
            or c.get('snapshot_revision') != tail.snapshot.name):
        raise ValueError('Pinned source capture required')
    reference = tail.replay(c['h40'], c['tokens'], c['S'])
    routes = {l: torch.cat(rows) for l, rows in tail.route_rows.items()}
    head = tail.logits(reference).argmax(-1)
    tail.backend = 'vnni'
    tail.cache.clear()
    adapted = tail.replay(c['h40'], c['tokens'], c['S'])
    same = torch.equal(reference.view(torch.int16), adapted.view(torch.int16))
    same_routes = all(torch.equal(routes[l], torch.cat(rows))
                      for l, rows in tail.route_rows.items())
    result = {
        'classification': 'CAPTURED_STATE_CPU_TAIL_ADAPTER_REFERENCE_PARITY',
        'production_certified': False, 'production_latency': None,
        'capture_sha256': hashlib.sha256(a.capture.read_bytes()).hexdigest(),
        'category': c['category'], 'positions': len(reference), 'S': c['S'],
        'cpu_adapter_h42_bit_equal': same, 'cpu_adapter_routes_equal': same_routes,
        'cpu_reference_vs_gpu_capture_h42_bit_equal': torch.equal(
            reference.view(torch.int16), c['h42'].view(torch.int16)),
        'cpu_reference_vs_gpu_capture_max_abs': float((reference.float() - c['h42'].float()).abs().max()),
        'cpu_reference_source_labels_equal': torch.equal(
            head[c['S'] - 1:], torch.tensor(c['tokens'][c['S']:], dtype=torch.long)),
        'distinct_routed_experts': {str(l): len(torch.unique(r)) for l, r in routes.items()},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    assert same and same_routes, 'CPU adapter parity failed; inspect saved result'


if __name__ == '__main__':
    main()
