"""Evaluate exported D3 trajectories through the pinned source tail; no timing claim."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from mosaicvm.bench.source_tail import SourceTail
from mosaicvm.bench.d1b_tail_robustness import prefix_metrics, prompt_mean_ci


def score(logits, tokens, S, horizons):
    labels = torch.tensor(tokens[S + 1:S + 1 + max(horizons)], dtype=torch.long)
    top = logits[S:S + len(labels)].topk(16, dim=-1).indices
    if len(labels) != max(horizons):
        raise ValueError('Missing terminal next-token label')
    hits = top[:, 0].eq(labels).tolist()
    result = {}
    for k in horizons:
        row = prefix_metrics(hits, k)
        row['topk_true_path_conditioned'] = {
            str(n): float(top[:k, :n].eq(labels[:k, None]).any(1).float().mean())
            for n in (1, 2, 4, 8, 16)}
        result[str(k)] = row
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--backend', choices=('reference', 'vnni'), default='reference')
    a = p.parse_args()
    files = sorted(a.predictions.glob('*.predictions.pt'))
    if not files:
        p.error('No prediction exports')
    torch.set_num_threads(2)
    tail = SourceTail(device=a.device, backend=a.backend)
    rows = []
    for file in files:
        c = torch.load(file, map_location='cpu', weights_only=True)
        if c['split'] != 'val' or c['snapshot_revision'] != tail.snapshot.name:
            raise ValueError('Expected validation data from pinned snapshot')
        S, K = c['S'], c['K']
        horizons = sorted({min(8, K), min(16, K), K})
        if c['h40_committed'].shape != (S, 4, 4096):
            raise ValueError('Bad exact committed prefix shape')
        exact = torch.cat([c['h40_committed'], c['h40_target']])
        y0 = tail.replay(exact, c['tokens'], S)
        lg0 = tail.logits(y0)
        target = torch.tensor(c['tokens'][S:S + K + 1], dtype=torch.long)
        bit_equal = torch.equal(y0.view(torch.int16), c['h42_target'].view(torch.int16))
        greedy_equal = torch.equal(lg0[S - 1:S + K].argmax(-1), target)
        if not bit_equal or not greedy_equal:
            raise RuntimeError(f'{c["category"]}: exact replay failed: h42={bit_equal}, head={greedy_equal}')
        for mode, pred in c['predicted_h40'].items():
            if pred.shape != (K, 4, 4096) or pred.dtype != torch.bfloat16:
                raise ValueError('Bad predicted trajectory shape/dtype')
            y = tail.replay(torch.cat([c['h40_committed'], pred]), c['tokens'], S)
            lg = tail.logits(y)
            if lg[S - 1].argmax() != target[0]:
                raise RuntimeError('Committed root changed')
            metrics = score(lg, c['tokens'], S, horizons)
            rows.append({'category': c['category'], 'mode': mode, 'split': 'val',
                         'exact_h42_bit_equal': bit_equal, 'exact_head_greedy_equal': greedy_equal,
                         'horizons': metrics, 'prediction_sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
            print(c['category'], mode, {k: v['E_A'] for k, v in metrics.items()}, flush=True)
        # Save completed prompts without presenting an unfinished aggregate as final.
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.with_suffix('.partial.json').write_text(json.dumps({'complete': False, 'rows': rows}, indent=2))
    aggregate = {}
    for mode in sorted({r['mode'] for r in rows}):
        mr = [r for r in rows if r['mode'] == mode]
        aggregate[mode] = {}
        for k in mr[0]['horizons']:
            values = [r['horizons'][k]['E_A'] for r in mr]
            aggregate[mode][k] = {
                'n_prompts': len(mr), 'E_A': sum(values) / len(values),
                'exploratory_prompt_ci95': prompt_mean_ci(values),
                'Sk': [sum(r['horizons'][k]['Sk'][i] for r in mr) / len(mr) for i in range(int(k))],
                'per_prompt_E_A': values}
    result = {'classification': 'D3_VALIDATION_SOURCE_TAIL_PREFIX_PROBE',
              'production_certified': False, 'student_pass': False,
              'geometry': 'K uncertain decisions exclude exact root; prefix+1',
              'conditioning': 'Reference-path inputs; prefix valid until first reject. Marginal Top-k after reject is true-path conditioned.',
              'recursive_chunk_feedback': 'predicted h40 only; no tail/token feedback',
              'device': a.device, 'expert_backend': a.backend,
              'source_tail_timing': None, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'rows': rows, 'aggregate': aggregate}
    a.output.write_text(json.dumps(result, indent=2))
    a.output.with_suffix('.partial.json').unlink(missing_ok=True)
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
