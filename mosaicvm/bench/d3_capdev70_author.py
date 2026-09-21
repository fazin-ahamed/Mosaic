"""Author CAP-DEV70 — fresh capability-promotion holdout.

70 prompts, 10 per domain, IDs {dom}_300..309. Val-only. Disjoint from
EVERY prior corpus (stage1/stage05, SCALE210/700/1000 train, SCALE-DEV3,
G1/G2 protected). Fail-closed on ID or normalized-text overlap.

This is the fresh exam for the frozen candidate HIST1001 + ps1001 + d3 B4.
Structurally distinct tasks per domain (not template x entity).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path("artifacts/d3_capdev70/prompts_capdev70.json")
AUDIT = Path("artifacts/d3_capdev70/overlap_audit.json")

EXISTING = [
    Path("artifacts/stage1_prompts.json"),
    Path("artifacts/stage05_prompts.json"),
    Path("artifacts/d3_scale/prompts_scale.json"),
    Path("artifacts/d3_scale700/prompts_scale700.json"),
    Path("artifacts/d3_scale1000/prompts_scale1000.json"),
    Path("artifacts/d3_scale1000/prompts_scale_dev3.json"),
]
PROTECTED = [
    Path("artifacts/routes_overlap_corpus_3060.json"),
    Path("artifacts/g2_cert_prompts.json"),
    Path("artifacts/g2_calibration_prompts.json"),
]

P = {
    "math": [
        "A regular hexagon has side length 4. What is its area in terms of the square root of three?",
        "How many integers between 1 and 200 are divisible by neither 4 nor 6?",
        "A geometric series has first term 3 and common ratio one half. What is its infinite sum?",
        "Two trains leave the same station going opposite directions at 60 and 90 km/h. How far apart are they after 40 minutes?",
        "What is the probability of rolling a sum of 8 with two fair six-sided dice?",
        "Factor the expression x squared minus 5x minus 14 completely.",
        "A cone has height 9 and base radius 4. What is its volume in terms of pi?",
        "The angles of a triangle are in the ratio 2 to 3 to 4. What is the largest angle in degrees?",
        "How many trailing zeros does 25 factorial have?",
        "A rectangle's length is twice its width and its perimeter is 54. What is its area?",
    ],
    "json": [
        "Output JSON for a wind turbine with turbine id, hub height meters, rated power kilowatts, and commissioning date.",
        "Create JSON for a comic issue with series title, issue number, cover artist, and cover price cents.",
        "Return JSON for a ferry crossing with vessel name, departure port, arrival port, and duration minutes.",
        "Generate JSON for a soil sample with plot id, pH value, nitrogen ppm, and sampling depth centimeters.",
        "Write JSON for a marathon runner with bib number, name, chip time seconds, and age group.",
        "Produce JSON for a warranty claim with claim id, product serial, fault code, and received-at date.",
        "Output JSON for a vineyard block with block name, grape variety, vine count, and planting year.",
        "Create JSON for a podcast episode with show name, episode number, duration seconds, and guest name.",
        "Return JSON for a blood donation with donor id, blood type, volume milliliters, and collected-at timestamp.",
        "Generate JSON for a satellite pass with satellite name, max elevation degrees, start azimuth, and pass duration seconds.",
    ],
    "retrieval": [
        "According to the quarry blast report, what charge weight was used in the eastern face?",
        "Which anchorage does the port pilot's boarding note assign to the inbound tanker?",
        "What curing time does the concrete pour record list for the bridge deck section?",
        "Who authorized the overtime shift in the bakery production ledger?",
        "What water pressure does the fire hydrant inspection card record for the corner unit?",
        "When does the aquarium feeding schedule say the seals are fed in the afternoon?",
        "How many pallets does the cold-store intake manifest log for the frozen berries?",
        "Which technician signed off the elevator annual load test in the maintenance book?",
        "Where does the campsite ranger's notice say to deposit food overnight?",
        "Which reagent did the brewery lab sheet flag as below specification?",
    ],
    "agent": [
        "Rename every file ending in .jpeg under photos to end in .jpg instead.",
        "Print the total size in bytes of all files modified in the last day under logs.",
        "Create a symbolic link named latest pointing at the newest file in releases.",
        "Show the first line of every .csv file in the reports folder with its filename.",
        "Set read-only permission on all files inside the archive directory.",
        "Find and print duplicate filenames that appear in both drafts and final.",
        "Write the current UTC timestamp into a file named heartbeat.txt every append.",
        "Truncate access.log to its last 500 lines in place.",
        "Print the line count of the largest .txt file under documents.",
        "Move all files older than a year from downloads into a folder named stale.",
    ],
    "coding": [
        "Write a function that returns the k most frequent elements of a list in descending frequency order.",
        "Write a function that checks whether two strings are anagrams ignoring spaces and case.",
        "Write a function that merges overlapping intervals given as a list of start-end pairs.",
        "Write a function that computes the edit distance between two strings.",
        "Write a function that rotates a square matrix 90 degrees clockwise in place.",
        "Write a function that returns the length of the longest increasing subsequence of a list.",
        "Write a function that validates whether a string of brackets is balanced across three bracket types.",
        "Write a function that flattens an arbitrarily nested list of integers into a single list.",
        "Write a function that finds the median of two sorted lists in logarithmic time.",
        "Write a function that generates all subsets of a list of distinct integers.",
    ],
    "longform": [
        "Explain how a heat pump moves warmth from cold outdoor air into a house.",
        "Describe how a lock on a canal raises a boat between two water levels.",
        "Explain how noise-cancelling headphones suppress ambient sound.",
        "Describe how a sourdough starter leavens bread without commercial yeast.",
        "Explain how a magnetic resonance imaging scanner produces an image of soft tissue.",
        "Describe how a windmill was used to drain water from low-lying polders.",
        "Explain how a refrigerator's compressor cycle keeps food cold.",
        "Describe how a suspension bridge carries its load down to the towers and anchors.",
        "Explain how a desalination plant removes salt from seawater by reverse osmosis.",
        "Describe how a glassblower shapes a molten gather into a finished vessel.",
    ],
    "convo": [
        "How do I ask a neighbor to trim a hedge that is blocking my driveway sightline?",
        "What should I consider before agreeing to pet-sit a friend's parrot for two weeks?",
        "How can I politely decline a coworker's offer to carpool without causing tension?",
        "What is a fair way to split a shared grocery bill when diets differ a lot?",
        "How do I bring up a recurring noise problem with an upstairs tenant calmly?",
        "What should I do if a delivery was left at the wrong apartment door?",
        "How can I stay focused during a long online workshop without drifting off?",
        "What is a tactful way to remind a friend they still owe me for concert tickets?",
        "How do I choose a reliable used bicycle without getting overcharged?",
        "What should I say to gracefully leave a party that is running much later than expected?",
    ],
}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _load_prompts(path: Path):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("prompts") or []


def re_stem(text: str) -> str:
    out = []
    for tok in text.lower().split():
        if any(ch.isdigit() for ch in tok):
            continue
        if "." in tok and not tok.endswith((".", "?", ",")):
            continue
        out.append(tok.strip(".,:;?'\""))
    return " ".join(out)


def collect_blocked():
    exist_ids, exist_texts, protected, sources = set(), set(), set(), {}
    for path in EXISTING:
        ids, texts = set(), set()
        for entry in _load_prompts(path):
            cat = entry.get("category") or entry.get("id")
            text = entry.get("prompt") or entry.get("text") or ""
            if cat:
                ids.add(cat); exist_ids.add(cat)
            if text:
                n = _norm(text); texts.add(n); exist_texts.add(n)
        sources[str(path)] = {"n_ids": len(ids), "n_texts": len(texts)}
    for path in PROTECTED:
        for entry in _load_prompts(path):
            text = entry.get("prompt") or entry.get("text") or ""
            if text:
                protected.add(_norm(text))
        sources[str(path)] = {"n_protected": len(protected)}
    return exist_ids, exist_texts, protected, sources


def main():
    exist_ids, exist_texts, protected, sources = collect_blocked()
    prompts, val = [], []
    id_hits, text_hits, prot_hits = [], [], []
    for dom, texts in P.items():
        assert len(texts) == 10 and len(set(texts)) == 10, dom
        for i, text in enumerate(texts):
            cat = f"{dom}_{300 + i:03d}"
            if cat in exist_ids:
                id_hits.append(cat)
            n = _norm(text)
            if n in exist_texts:
                text_hits.append({"category": cat, "prompt": text})
            if n in protected:
                prot_hits.append({"category": cat, "prompt": text})
            prompts.append({"category": cat, "prompt": text, "domain": dom})
            val.append(cat)

    assert not id_hits, f"ID overlap: {id_hits}"
    assert not text_hits, f"text overlap: {text_hits}"
    assert not prot_hits, f"protected overlap: {prot_hits}"
    assert len(val) == 70 and len(prompts) == 70
    assert len({_norm(p["prompt"]) for p in prompts}) == 70
    assert len({p["category"] for p in prompts}) == 70
    for dom, texts in P.items():
        stems = [re_stem(t) for t in texts]
        assert len(set(stems)) == 10, (dom, stems)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "classification": "STAGE1_CAP_DEV70",
        "purpose": "CAP-DEV70 fresh capability-promotion holdout; "
                   "val-only; disjoint from all train + SCALE-DEV3",
        "splits": {"train": [], "val": val},
        "prompts": prompts,
        "author_note": "IDs 300-309; 10 structurally distinct "
                       "tasks/domain; frozen before candidate eval",
    }
    OUT.write_text(json.dumps(body, indent=1) + "\n")
    man_sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    src_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit = {
        "classification": "CAP_DEV70_OVERLAP_AUDIT", "n": 70, "ids": val,
        "blocked_sources": sources, "id_overlap": id_hits,
        "text_overlap": text_hits, "protected_overlap": prot_hits,
        "all_empty_required": True, "manifest_sha256": man_sha,
        "author_sha256": src_sha,
    }
    AUDIT.write_text(json.dumps(audit, indent=1) + "\n")
    print(f"wrote {OUT}: 0 train + {len(val)} val")
    print("manifest sha256:", man_sha)
    print("overlap checks empty:",
          not id_hits and not text_hits and not prot_hits)


if __name__ == "__main__":
    main()
