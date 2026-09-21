"""Author SCALE-DEV3 (roadmap SCALE1000 §12–§14).

70 new holdout prompts, 10 per domain, IDs {dom}_127..{dom}_136.
Val-only: no SCALE1000 train texts in this file.

Guards (fail-closed):
  ID overlap against stage1, stage05, SCALE210, SCALE700
  normalized text overlap against those plus G1/G2
  10 structurally distinct tasks per domain (not template x entity)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path("artifacts/d3_scale1000/prompts_scale_dev3.json")
AUDIT = Path("artifacts/d3_scale1000/overlap_audit_dev3.json")

EXISTING = [
    Path("artifacts/stage1_prompts.json"),
    Path("artifacts/stage05_prompts.json"),
    Path("artifacts/d3_scale/prompts_scale.json"),
    Path("artifacts/d3_scale700/prompts_scale700.json"),
]
PROTECTED = [
    Path("artifacts/routes_overlap_corpus_3060.json"),
    Path("artifacts/g2_cert_prompts.json"),
    Path("artifacts/g2_calibration_prompts.json"),
]

# Ten structurally distinct tasks per domain. Not a shared
# template filled with ten entities.
P = {
    "math": [
        "How many distinct 5-card hands can be dealt from a 52-card deck?",
        "A clock's minute hand is on 12. How many degrees does it travel in 18 minutes?",
        "What is the remainder when 2 to the power of 15 is divided by 7?",
        "A sphere has radius 6. What is its surface area in terms of pi?",
        "If the mean of five numbers is 12 and four of them are 8, 10, 14, and 16, what is the fifth?",
        "Convert 3 hours and 27 minutes into seconds.",
        "A ladder 13 meters long leans against a wall with its base 5 meters from the wall. How high does it reach?",
        "What is the 12th triangular number?",
        "A store marks an item up 25 percent from 80 dollars, then takes 10 percent off the marked price. What is the final price?",
        "Solve for n: the sum of the first n odd numbers equals 121.",
    ],
    "json": [
        "Output JSON for a shipping carton with tracking number, origin warehouse, destination postal code, and kilogram weight.",
        "Create JSON for a museum exhibit with accession number, artist, medium, and year acquired.",
        "Return JSON for a bus route with route id, stop names array, weekday start time, and fare in cents.",
        "Generate JSON for a laboratory sample with barcode, collected-at timestamp, technician id, and storage temperature.",
        "Write JSON for a chess game with white player, black player, PGN moves array, and result string.",
        "Produce JSON for a parking citation with ticket number, license plate, violation code, and issued-at datetime.",
        "Output JSON for a radio station with call sign, frequency megahertz, city of license, and format.",
        "Create JSON for a hiking trail with trail name, length kilometers, elevation gain meters, and difficulty.",
        "Return JSON for a vaccine lot with lot number, manufacturer, expiration date, and doses remaining.",
        "Generate JSON for a telescope observation with target name, right ascension, declination, and exposure seconds.",
    ],
    "retrieval": [
        "According to the harbor master's log, which berth was assigned to the arriving ferry?",
        "What moisture threshold does the grain silo inspection sheet list for safe storage?",
        "Who countersigned the night-shift handover in the ward nursing notes?",
        "Which fuse rating does the substation switching order specify for feeder B?",
        "When does the concert hall house-manager brief say the lobby doors open?",
        "How many spare rails does the track-maintenance work order allocate to milepost 14?",
        "Why was the orchard frost-protection protocol activated according to the agronomy diary?",
        "What visitor-capacity limit is printed on the observatory public-night flyer?",
        "Which reagent lot failed the titration check in the quality-control worksheet?",
        "Where does the lighthouse keepers' standing order say to store the spare lamp?",
    ],
    "agent": [
        "Print the inode number of kernel.img.",
        "Compute the SHA-256 digest of firmware.bin and write it to firmware.sha256.",
        "List every empty subdirectory under the folder named scratch.",
        "Change the owner of secrets.env to the user named deploy.",
        "Show which process currently has the file lock.sock open.",
        "Create a 50-megabyte sparse file called scratch.img.",
        "Compare byte-for-byte the files gold.wav and candidate.wav and print whether they match.",
        "Set the modification time of snapshot.raw to midnight UTC on 1 January 2020.",
        "Extract only the files ending in .po from locale.tgz into a folder named translations.",
        "Count how many hard links point at the inode of canonical.dat.",
    ],
    "coding": [
        "Write a function that returns the longest palindromic substring of a string.",
        "Write a function that evaluates a postfix arithmetic expression given as a list of tokens.",
        "Write a function that inserts a value into a max-heap represented as an array and restores the heap property.",
        "Write a function that converts an integer to its Roman-numeral representation for values 1 through 3999.",
        "Write a function that finds a peak element in an unsorted list, defined as an element not smaller than its neighbors.",
        "Write a function that serializes a binary tree to a string using preorder with null markers and deserializes it back.",
        "Write a function that computes the minimum number of coins of given denominations needed to make a target amount.",
        "Write a function that returns the next permutation of a list of integers in lexicographic order.",
        "Write a function that detects whether a directed graph given as an adjacency list contains a cycle.",
        "Write a function that compresses a string with run-length encoding and expands the encoded form back to the original.",
    ],
    "longform": [
        "Explain how a seismograph records ground motion during an earthquake.",
        "Describe how a blast furnace turns iron ore into pig iron.",
        "Explain how a radio telescope forms an image of a distant source.",
        "Describe how a beehive super is harvested without destroying the colony.",
        "Explain how a catalytic converter reduces exhaust pollutants.",
        "Describe how a tide mill captured energy from rising and falling water.",
        "Explain how a vacuum flask keeps its contents hot or cold.",
        "Describe how a papermaking Fourdrinier machine forms a continuous sheet.",
        "Explain how a gyrocompass finds true north without a magnetic needle.",
        "Describe how a municipal composting windrow is turned and monitored.",
    ],
    "convo": [
        "How do I tell a roommate I need the shared bathroom first on work mornings?",
        "What should I check before lending a neighbor my spare folding chairs?",
        "How can I keep a long train delay from ruining a tight connection?",
        "What is a fair way to rotate who chooses the family movie night?",
        "How do I ask a dentist to explain a treatment plan in plain language?",
        "What should I do if I find a wallet with no ID on a park bench?",
        "How can I practice a speech without annoying people in the next room?",
        "What is a considerate way to cancel a standing weekly game night?",
        "How do I choose a raincoat that will actually keep me dry in wind?",
        "What should I say when a shop clerk has already started wrapping the wrong size?",
    ],
}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _load_prompts(path: Path):
    data = json.loads(path.read_text())
    return data.get("prompts") or []


def collect_blocked():
    exist_ids, exist_texts = set(), set()
    sources = {}
    for path in EXISTING:
        ids, texts = set(), set()
        for entry in _load_prompts(path):
            cat = entry.get("category") or entry.get("id")
            text = entry.get("prompt") or entry.get("text") or ""
            if cat:
                ids.add(cat)
                exist_ids.add(cat)
            if text:
                n = _norm(text)
                texts.add(n)
                exist_texts.add(n)
        sources[str(path)] = {"n_ids": len(ids), "n_texts": len(texts)}
    protected = set()
    for path in PROTECTED:
        for entry in _load_prompts(path):
            text = entry.get("prompt") or entry.get("text") or ""
            if text:
                protected.add(_norm(text))
        sources[str(path)] = {"n_protected_texts": len(protected)}
    return exist_ids, exist_texts, protected, sources


def main():
    exist_ids, exist_texts, protected, sources = collect_blocked()
    prompts, val = [], []
    id_hits, text_hits, prot_hits = [], [], []
    for dom, texts in P.items():
        assert len(texts) == 10, (dom, len(texts))
        assert len(set(texts)) == 10, (dom, "internal dup")
        for i, text in enumerate(texts):
            cat = f"{dom}_{127 + i:03d}"
            if cat in exist_ids:
                id_hits.append(cat)
            n = _norm(text)
            if n in exist_texts:
                text_hits.append({"category": cat, "prompt": text})
            if n in protected:
                prot_hits.append({"category": cat, "prompt": text})
            prompts.append({"category": cat, "prompt": text,
                            "domain": dom})
            val.append(cat)

    assert not id_hits, f"ID overlap: {id_hits}"
    assert not text_hits, f"text overlap: {text_hits}"
    assert not prot_hits, f"protected overlap: {prot_hits}"
    assert len(val) == 70 and len(prompts) == 70
    norms = [_norm(p["prompt"]) for p in prompts]
    assert len(set(norms)) == 70
    assert len({p["category"] for p in prompts}) == 70

    # structural: each domain's 10 stems must differ after
    # stripping numbers / quoted filenames
    for dom, texts in P.items():
        stems = []
        for t in texts:
            s = re_stem(t)
            stems.append(s)
        assert len(set(stems)) == 10, (dom, stems)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "classification": "STAGE1_SCALE_DEV3",
        "purpose": "SCALE-DEV3 fresh holdout for SCALE1000; "
                   "val-only; no train additions",
        "splits": {"train": [], "val": val},
        "prompts": prompts,
        "author_note": "IDs 127-136; 10 structurally distinct "
                       "tasks/domain; frozen before SCALE1000 train",
    }
    OUT.write_text(json.dumps(body, indent=1) + "\n")
    man_sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    src_sha = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    audit = {
        "classification": "SCALE_DEV3_OVERLAP_AUDIT",
        "n": 70,
        "ids": val,
        "blocked_sources": sources,
        "id_overlap": id_hits,
        "text_overlap": text_hits,
        "protected_overlap": prot_hits,
        "all_empty_required": True,
        "manifest_sha256": man_sha,
        "author_sha256": src_sha,
    }
    AUDIT.write_text(json.dumps(audit, indent=1) + "\n")
    print(f"wrote {OUT}: 0 train + {len(val)} val")
    print("manifest sha256:", man_sha)
    print("author sha256:", src_sha)
    print("audit sha256:", hashlib.sha256(
        AUDIT.read_bytes()).hexdigest())
    print("overlap checks empty:",
          not id_hits and not text_hits and not prot_hits)


def re_stem(text: str) -> str:
    """Cheap structural fingerprint: drop digits and isolated
    filenames so template x entity clones collapse."""
    out = []
    for tok in text.lower().split():
        if any(ch.isdigit() for ch in tok):
            continue
        if "." in tok and not tok.endswith((".", "?", ",")):
            continue
        out.append(tok.strip(".,:;?'\""))
    return " ".join(out)


if __name__ == "__main__":
    main()
