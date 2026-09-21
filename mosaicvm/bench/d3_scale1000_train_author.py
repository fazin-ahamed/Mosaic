"""Author SCALE1000 train additions (roadmap §15–§20).

301 new training prompts, 43 per domain, IDs {dom}_137..{dom}_179.
Loads frozen SCALE-DEV3; never edits it.

Combined capture manifest:
  val  = SCALE-DEV3 70 prompts ({dom}_127..136)
  train = these 301 prompts ({dom}_137..179)

Guards (fail-closed): ID + normalized-text overlap against
stage1, stage05, SCALE210, SCALE700, SCALE-DEV3, G1/G2.
43 structurally distinct tasks per domain.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

DEV3 = Path("artifacts/d3_scale1000/prompts_scale_dev3.json")
OUT = Path("artifacts/d3_scale1000/prompts_scale1000.json")
AUDIT = Path("artifacts/d3_scale1000/overlap_audit_train.json")

EXISTING = [
    Path("artifacts/stage1_prompts.json"),
    Path("artifacts/stage05_prompts.json"),
    Path("artifacts/d3_scale/prompts_scale.json"),
    Path("artifacts/d3_scale700/prompts_scale700.json"),
    DEV3,
]
PROTECTED = [
    Path("artifacts/routes_overlap_corpus_3060.json"),
    Path("artifacts/g2_cert_prompts.json"),
    Path("artifacts/g2_calibration_prompts.json"),
]

# Frozen DEV3 sha recorded at freeze time. Fail if it moved.
DEV3_SHA = (
    "41c5b929560e4605cb66ce139fef761b136d4dad"
    "7be3c0fbddac4ac18718eacf"
)

P = {
    "math": [
        "How many diagonals does a convex 14-sided polygon have?",
        "A wheel of radius 35 centimeters rolls 11 meters. How many full revolutions does it make?",
        "What is the least common multiple of 18, 24, and 30?",
        "A right triangle has legs 9 and 12. What is the length of the altitude to the hypotenuse?",
        "If 3 workers finish a wall in 8 days, how many days do 6 workers need at the same rate?",
        "Convert 2.4 radians to degrees, rounded to the nearest degree.",
        "What is the sum of the interior angles of a convex heptagon, in degrees?",
        "A geometric sequence starts 5, 15, 45. What is the 6th term?",
        "How many trailing zeros does 40 factorial end with?",
        "A circle is inscribed in a square of side 10. What is the area of the region inside the square but outside the circle, in terms of pi?",
        "Solve for x: log base 3 of (x plus 6) equals 2.",
        "A bag holds 4 red and 6 blue marbles. Two are drawn without replacement. What is the probability both are red?",
        "What is the continued-fraction convergent of the square root of 2 after three steps, as a reduced fraction?",
        "A 5 by 5 grid of points. How many squares of any size have vertices on the grid points?",
        "If sin theta equals 3/5 and theta is acute, what is tan theta?",
        "How many integers between 1 and 200 are divisible by 3 or 5 but not both?",
        "A cone has height 12 and base radius 5. What is its volume in terms of pi?",
        "The arithmetic mean of 7 consecutive integers is 15. What is the largest of them?",
        "What is 17 in base 2, written without a prefix?",
        "A clock loses 4 minutes per hour. How many real minutes pass before it shows a 1-hour loss?",
        "Find the smallest prime that is 1 greater than a perfect square and 1 less than a perfect cube.",
        "A rectangle has perimeter 54 and area 180. What are its side lengths?",
        "How many distinct real roots does x cubed minus 3x plus 2 equal zero have?",
        "A 12-liter mixture is 25 percent juice. How many liters of pure juice must be added to make it 40 percent juice?",
        "What is the 8th tetrahedral number?",
        "Two trains 120 kilometers apart approach at 50 and 70 kilometers per hour. When do they meet, in hours?",
        "Convert 5 acres to square meters using 1 acre equals 4047 square meters.",
        "A regular octagon has side 4. What is its area in simplest radical form?",
        "How many ways can 8 identical chairs be placed around a circular table if rotations count as the same?",
        "If f of x equals 2x plus 3 and g of x equals x squared, what is g of f of 2?",
        "A rhombus has diagonals 10 and 24. What is its side length?",
        "What is the units digit of 7 to the power of 2026?",
        "A tank fills through pipe A in 6 hours and empties through pipe B in 8 hours. If both are open, how long to fill an empty tank?",
        "Find the determinant of the 2 by 2 matrix with rows 4, 7 and 3, 5.",
        "How many lattice points lie strictly inside the circle x squared plus y squared equals 25?",
        "A 20 percent salt solution is mixed with a 50 percent solution to make 12 liters of 30 percent solution. How many liters of the 20 percent solution are used?",
        "What is the harmonic mean of 6 and 12?",
        "A pentagon has exterior angles all equal. What is each exterior angle in degrees?",
        "If the product of two consecutive even integers is 288, what is the pair?",
        "A cylindrical can has radius 4 and height 10. What is its total surface area in terms of pi?",
        "How many 4-digit numbers have all distinct digits and do not start with zero?",
        "A pendulum 1 meter long swings through a 30-degree arc. How long is the path of the bob, in terms of pi?",
        "What is the remainder when 5 to the 8th is divided by 13?",
    ],
    "json": [
        "Output JSON for a ferry crossing with vessel name, departure quay, arrival quay, and scheduled UTC time.",
        "Create JSON for a weather balloon launch with flight id, burst altitude meters, gas type, and payload grams.",
        "Return JSON for a courtroom docket entry with case number, judge, hearing type, and scheduled date.",
        "Generate JSON for a greenhouse bench with bench id, crop, planting date, and target humidity percent.",
        "Write JSON for a lighthouse watch with station name, keeper, watch start, and visibility nautical miles.",
        "Produce JSON for a quarry blast with shot number, explosive type, delay milliseconds, and peak particle velocity.",
        "Output JSON for a dairy vat with vat id, milk liters, culture strain, and set temperature celsius.",
        "Create JSON for a tram stop with stop code, line names array, shelter present, and next-departure ISO time.",
        "Return JSON for a forest plot survey with plot id, species counts object, canopy percent, and surveyor.",
        "Generate JSON for a diving log with dive number, maximum depth meters, bottom time minutes, and gas mix.",
        "Write JSON for a foundry pour with heat number, alloy, pour temperature, and mold id.",
        "Produce JSON for a canal lockage with lock name, vessel, direction, and water-level change meters.",
        "Output JSON for a beehive inspection with hive id, frames of brood, mite count, and inspector.",
        "Create JSON for a radio telescope scan with scan id, center frequency megahertz, bandwidth, and integration seconds.",
        "Return JSON for a bakery oven cycle with oven id, product, bake minutes, and steam on boolean.",
        "Generate JSON for a glacier stake with stake id, measured ablation centimeters, date, and observer.",
        "Write JSON for a customs declaration with declaration id, HS code, origin country, and mass kilograms.",
        "Produce JSON for a wind-farm turbine with turbine id, hub height meters, rotor diameter, and rated megawatts.",
        "Output JSON for a choir rehearsal with ensemble, venue, start time, and repertoire titles array.",
        "Create JSON for a soil core with core id, depth centimeters, pH, and organic-matter percent.",
        "Return JSON for a tugboat job with job id, assisted vessel, bollard pull tonnes, and harbor.",
        "Generate JSON for a print-shop job with job ticket, paper stock, color profile, and copy count.",
        "Write JSON for a veterinary visit with patient microchip, species, procedure, and attending vet.",
        "Produce JSON for a ski-lift run with lift name, chairs occupied, wind speed, and hold boolean.",
        "Output JSON for a sewage pump station with station id, inflow liters per second, wet-well level, and alarm state.",
        "Create JSON for a manuscript folio with shelfmark, folio number, script type, and digitized URI.",
        "Return JSON for a crane lift with lift plan id, load kilograms, radius meters, and exclusion-zone radius.",
        "Generate JSON for a tidal gauge reading with station, timestamp UTC, water level meters, and datum.",
        "Write JSON for a seed packet lot with lot code, cultivar, germination percent, and packed-on date.",
        "Produce JSON for a fire-watch tower log with tower, observer, smoke bearing degrees, and distance estimate.",
        "Output JSON for a subway car inspection with car number, odometer kilometers, defect codes array, and inspector.",
        "Create JSON for a vineyard block with block id, varietal, trellis type, and hectares.",
        "Return JSON for a hospital linen cart with cart id, ward, item counts object, and dispatched-at.",
        "Generate JSON for a meteor report with observer, peak magnitude, radiant, and duration seconds.",
        "Write JSON for a dry-dock booking with vessel, dock name, keel-block plan id, and start date.",
        "Produce JSON for a classroom occupancy with room, period, enrolled count, and present count.",
        "Output JSON for a hydrophone drop with drop id, latitude, longitude, and cable meters.",
        "Create JSON for a clock-repair ticket with movement caliber, complication, parts used array, and due date.",
        "Return JSON for a compost windrow with row id, temperature celsius, moisture percent, and turn date.",
        "Generate JSON for a pipeline pig run with pig id, launch station, receive station, and expected hours.",
        "Write JSON for a museum loan with loan number, object accession, borrower, and return date.",
        "Produce JSON for a rooftop apiary with hive count, building address, last inspection, and contact.",
        "Output JSON for a glacier weather station with station id, air temperature, wind gust, and snow depth centimeters.",
    ],
    "retrieval": [
        "According to the dry-dock survey, which hull plate was marked for renewal?",
        "What minimum cable tension does the aerial-tramway inspection card list?",
        "Who authorized the after-hours vault opening in the museum security blotter?",
        "Which lot number does the seed-lab germination sheet reject?",
        "When does the ice-road operations bulletin close the crossing to trucks?",
        "How many spare impellers does the pump-house spare-parts list assign to station 3?",
        "Why was the apiary quarantine extended according to the state apiarist memo?",
        "What occupancy cap is printed on the planetarium dome-show run sheet?",
        "Which weld procedure failed the radiograph review in the pressure-vessel dossier?",
        "Where does the lock-keeper's standing order say to stow the spare sluice crank?",
        "According to the avalanche-control log, which slope received the morning charges?",
        "What dissolved-oxygen floor does the hatchery water-quality sheet require?",
        "Who signed the night-watch turnover in the grain-elevator logbook?",
        "Which fuse did the traction-substation switching sheet pull for track 2?",
        "When does the ferry-terminal traffic plan open the upper car deck?",
        "How many extra sleepers does the rail-relay work order send to milepost 9?",
        "Why was the orchard hail-net protocol skipped according to the field diary?",
        "What visitor limit is on the cave-tour safety card for chamber B?",
        "Which reagent expired on the titration bench per the chemistry stock card?",
        "Where does the range-safety order store the misfire kit?",
        "According to the dredge report, what depth was reached at buoy 12?",
        "What wind-hold threshold does the crane-lift permit list in knots?",
        "Who countersigned the blood-bank temperature excursion form?",
        "Which transformer tap is specified on the mill switchgear order?",
        "When does the concert-hall fire-curtain test memo schedule the drop?",
        "How many spare sheaves does the ski-lift maintenance sheet allocate to tower 7?",
        "Why was the compost-windrow turn delayed according to the operations note?",
        "What capacity is printed on the observatory visitor-night ticket?",
        "Which batch failed the moisture check on the malt-house kiln sheet?",
        "Where does the lightship standing order keep the spare lens cloths?",
        "According to the quarry shot-plan, which delay sequence was used on bench 4?",
        "What brine concentration does the cheese-vat log require at salting?",
        "Who approved the after-dark runway inspection in the airfield log?",
        "Which cable drum is listed as spare on the tramway stores card?",
        "When does the lock-flight timetable close the chamber for maintenance?",
        "How many extra fenders does the harbor-master order send to berth 6?",
        "Why was the greenhouse shade-cloth retracted according to the grower note?",
        "What load limit is stenciled on the canal-bridge rating plate?",
        "Which culture tube was discarded on the dairy-lab worksheet?",
        "Where does the fire-watch standing order hang the spare binoculars?",
        "According to the glacier-stake sheet, which stake showed the largest ablation?",
        "What minimum visibility does the harbor fog-signal instruction require before reducing speed?",
        "Who initialed the night-shift boiler blowdown in the plant log?",
    ],
    "agent": [
        "Print the SELinux context of boot.img.",
        "Compute the BLAKE2b digest of payload.iso and write it to payload.b2.",
        "List every directory under var/spool that contains no files.",
        "Change the group of keys.pem to the group named ops.",
        "Show which process holds an exclusive flock on spool.lock.",
        "Create a 200-megabyte sparse file named scratch.raw.",
        "Compare byte-for-byte master.flac and copy.flac and exit nonzero if they differ.",
        "Set the access time of archive.cpio to noon UTC on 4 July 2012.",
        "Extract only files ending in .mo from messages.tar into a folder named locale.",
        "Count how many names in the current directory are symbolic links.",
        "Print the UUID of the filesystem that contains root.img.",
        "Write a 4096-byte file of zeroes called pad.bin using dd.",
        "Show the default ACL on shared.dir if any is set.",
        "Rename every file ending in .jpeg under photos/ to .jpg without touching directories.",
        "Print the major and minor device numbers of loop0.",
        "Create a named pipe called ingest.fifo in /tmp.",
        "Show the mount options of the filesystem that holds cache.db.",
        "Strip the execute bit from every file in scripts/ that is not a directory.",
        "Print the number of 512-byte blocks allocated to huge.dat.",
        "Copy sparse.img to sparse2.img preserving holes.",
        "List open file descriptors of the process whose pid is in run.pid.",
        "Write the current UNIX epoch to stamp.txt as a single integer.",
        "Show whether core.core is a core dump using the file command.",
        "Change the permissions of private.key to 0400.",
        "Print the target of the symlink named current.",
        "Create an empty directory tree a/b/c/d with a single command.",
        "Show the birth time of notes.adoc if the filesystem records it.",
        "Concatenate part.aa through part.ad in order into restore.bin.",
        "Print the inode generation number of canonical.idx if the tool supports it.",
        "Zero the first 512 bytes of disk.img without truncating the rest.",
        "List every file under logs/ that has not been modified in 30 days.",
        "Show the supplementary groups of the owner of cron.tab.",
        "Create a relative symlink named latest pointing at build-42.",
        "Print the CRC32 of firmware.hex.",
        "Move all files matching *.bak from the current directory into backup/ creating the folder if needed.",
        "Show whether socket.sock is a UNIX domain socket.",
        "Write a one-line fstab entry for data.img mounted at /mnt/data to stdout.",
        "Print the number of hard links to inode 2 of the current filesystem.",
        "Set the immutable attribute on sealed.dat if the filesystem allows it.",
        "Show the SELinux user of process 1.",
        "Create a 1-kilobyte file named tiny.bin filled with the byte 0xFF.",
        "Print the parent directory of the real path of this.lnk.",
        "Count the number of regular files under tree/ that are empty.",
    ],
    "coding": [
        "Write a function that returns the kth smallest element in an unsorted list using a linear-time selection algorithm.",
        "Write a function that multiplies two polynomials given as coefficient lists and returns the product coefficients.",
        "Write a function that finds the lowest common ancestor of two nodes in a binary tree given parent pointers.",
        "Write a function that converts a postfix expression to an infix expression with minimal parentheses.",
        "Write a function that computes the edit distance between two strings with insert, delete, and substitute cost 1.",
        "Write a function that reconstructs a binary tree from its inorder and postorder traversals.",
        "Write a function that returns all subsets of a list whose elements sum to a target, without duplicate subsets.",
        "Write a function that finds the longest path in a weighted directed acyclic graph given as an adjacency list.",
        "Write a function that implements a circular buffer of fixed capacity with enqueue, dequeue, and peek.",
        "Write a function that checks whether a 9 by 9 board is a valid Sudoku, ignoring empty cells.",
        "Write a function that computes the convex hull of a list of 2D points and returns the hull vertices in order.",
        "Write a function that evaluates an arithmetic expression with plus, minus, times, and parentheses given as a string.",
        "Write a function that finds a majority element appearing more than n over 2 times, or reports that none exists.",
        "Write a function that transposes a sparse matrix stored as a list of (row, column, value) triples.",
        "Write a function that generates the next Gray-code integer after a given nonnegative integer.",
        "Write a function that returns the maximum area of water that can be held between two lines in a height list.",
        "Write a function that implements union-find with path compression and union by rank, exposing find and union.",
        "Write a function that converts a Unix timestamp in seconds to an ISO-8601 UTC string without a timezone library.",
        "Write a function that finds all starting indices of anagrams of a word inside a longer string.",
        "Write a function that computes the shortest superstring of a list of strings using a greedy overlap heuristic.",
        "Write a function that returns the median of two already-sorted lists in logarithmic time.",
        "Write a function that simulates a simple stack machine with push, pop, add, and mul instructions.",
        "Write a function that determines whether parentheses, brackets, and braces in a string are correctly nested.",
        "Write a function that packs a list of integers into a bitset represented as a Python int and unpacks it.",
        "Write a function that finds the longest common subsequence of two strings and returns one such subsequence.",
        "Write a function that interpolates a missing value in a list of numbers using linear interpolation between neighbors.",
        "Write a function that topologically sorts a directed acyclic graph or raises if a cycle exists.",
        "Write a function that computes the Hamming distance between two equal-length bit strings.",
        "Write a function that rotates an n by n matrix 90 degrees clockwise in place.",
        "Write a function that parses a simple CSV line handling quoted fields with embedded commas.",
        "Write a function that returns the sliding-window maximum of a list for a given window size k.",
        "Write a function that converts RGB triples to grayscale using the rec.601 luma coefficients.",
        "Write a function that finds an Eulerian path in an undirected graph if one exists.",
        "Write a function that implements a min-stack supporting push, pop, top, and get-min in constant time.",
        "Write a function that counts inversions in a list using a modified merge sort.",
        "Write a function that validates an IPv6 address in compressed form including the double-colon shorthand.",
        "Write a function that computes the discrete convolution of two equal-length real lists.",
        "Write a function that finds the first missing positive integer in an unsorted list containing negatives.",
        "Write a function that renders a spiral matrix of size n filled with 1 through n squared.",
        "Write a function that decodes a run of Morse code letters separated by spaces, using a provided mapping.",
        "Write a function that returns the longest increasing subsequence of a list of integers, not just its length.",
        "Write a function that checks whether two binary trees are mirrors of each other.",
        "Write a function that computes the checksum of a byte string as the two's-complement of the sum of bytes modulo 256.",
    ],
    "longform": [
        "Explain how a Foucault pendulum demonstrates Earth's rotation.",
        "Describe how a Bessemer converter removed carbon from molten iron.",
        "Explain how a maser amplifies microwave signals.",
        "Describe how a cooper shapes and hoops a wooden barrel.",
        "Explain how a Stirling engine converts a temperature difference into motion.",
        "Describe how a charcoal clamp is built and burned.",
        "Explain how a planimeter measures the area of an irregular plane figure.",
        "Describe how a dry-stone wall is laid so it stands without mortar.",
        "Explain how a siphon transfers liquid over a barrier.",
        "Describe how a watermill's launder and pentrough feed the wheel.",
        "Explain how a thermocouple produces a voltage from two metals.",
        "Describe how a glassblower gathers, blows, and anneals a simple vessel.",
        "Explain how a differential gear lets driven wheels rotate at different speeds.",
        "Describe how a salt pan is flooded and harvested.",
        "Explain how a Geiger-Muller tube detects ionizing radiation.",
        "Describe how a thatcher lays a reed roof.",
        "Explain how a flywheel stores rotational energy.",
        "Describe how a lime kiln converts limestone into quicklime.",
        "Explain how a Bourdon tube gauge measures pressure.",
        "Describe how a miller dresses a millstone.",
        "Explain how a thermoelectric cooler moves heat with a Peltier junction.",
        "Describe how a rigger sets standing and running rigging on a small sloop.",
        "Explain how a choke point in a river is used to measure discharge with a weir.",
        "Describe how a coppersmith raises a bowl from sheet metal.",
        "Explain how a gyroscopic stabilizer resists roll on a ship.",
        "Describe how a peat cutter dries and stacks fuel turves.",
        "Explain how a strain gauge changes resistance under load.",
        "Describe how a bell founder tunes a bell after casting.",
        "Explain how a heat pipe moves heat with a working fluid and wick.",
        "Describe how a hopback separates wort from hops in a brewery.",
        "Explain how a vernier caliper interpolates between scale marks.",
        "Describe how a wheelwright shrinks an iron tyre onto a wooden wheel.",
        "Explain how a Tesla coil produces high-voltage high-frequency discharges.",
        "Describe how a tanner converts hide into leather in a vegetable pit.",
        "Explain how a pitot tube infers airspeed from pressure difference.",
        "Describe how a charcoal iron is filled and used for pressing cloth.",
        "Explain how a Wheatstone bridge measures an unknown resistance.",
        "Describe how a millwright levels and keys a line shaft.",
        "Explain how a solenoid valve opens a fluid path with an electromagnet.",
        "Describe how a cider maker mills and presses apples.",
        "Explain how a bimetallic strip bends with temperature.",
        "Describe how a farrier forges and nails a horseshoe.",
        "Explain how a regenerative heat exchanger recaptures exhaust heat.",
    ],
    "convo": [
        "How do I ask a houseguest to take shorter showers without sounding hostile?",
        "What should I check before borrowing a neighbor's folding table for a potluck?",
        "How can I keep a delayed bus from making me miss a timed museum entry?",
        "What is a fair way to rotate who cooks on a three-person weeknight schedule?",
        "How do I ask a pharmacist to explain a generic substitution in plain words?",
        "What should I do if I find a labeled set of keys on a cafe windowsill?",
        "How can I rehearse a toast without keeping the whole house awake?",
        "What is a considerate way to skip a monthly book club without quitting?",
        "How do I choose walking shoes that will not blister on cobblestones?",
        "What should I say when a cashier has already bagged an item I want to put back?",
        "How do I tell a coworker their desk fan is blowing papers onto my keyboard?",
        "What should I bring to a picnic if I cannot keep food cold for two hours?",
        "How can I split a taxi fare when three people got off at different stops?",
        "What is a kind way to ask a relative to stop posting photos of my child?",
        "How do I pack a carry-on so a laptop comes out without unpacking clothes?",
        "What should I do if a restaurant bills me for a dish I sent back?",
        "How can I keep a borrowed jacket from smelling like my cologne?",
        "What is a polite way to leave a group chat that is only about a trip I skipped?",
        "How do I choose a bicycle lock that will actually hold in a busy station?",
        "What should I say when a host offers a second helping I cannot finish?",
        "How do I ask building management to fix a buzzing hallway light?",
        "What should I check before watering a neighbor's plants while they travel?",
        "How can I keep a long layover from wrecking my sleep the next day?",
        "What is a fair way to share a driveway when two households park at odd hours?",
        "How do I ask a barber to match a photo without sounding picky?",
        "What should I do if I receive mail for a previous tenant with a check inside?",
        "How can I practice piano scales in an apartment without rattling the floor?",
        "What is a considerate way to decline being tagged in a large group photo?",
        "How do I choose a suitcase that will stay under a budget airline's weight limit?",
        "What should I say when a shop has already cut fabric longer than I asked?",
        "How do I tell a roommate the shared milk is gone without starting a tally war?",
        "What should I pack for a day hike if the forecast flips from sun to rain at noon?",
        "How can I return a borrowed dish the day after a party without seeming abrupt?",
        "What is a kind way to ask someone to take their call outside a waiting room?",
        "How do I keep a bicycle chain from staining work trousers on a commute?",
        "What should I do if a delivery driver leaves a package in the rain?",
        "How can I split leftovers fairly after a potluck at someone else's house?",
        "What is a polite way to ask a guest to smoke away from an open window?",
        "How do I choose earplugs that will still let me hear a smoke alarm?",
        "What should I say when a colleague uses my mug from the office cupboard?",
        "How do I ask a landlord about a slow drain without sounding like I caused it?",
        "What should I check before feeding a neighbor's cat wet food for the first time?",
        "How can I keep a surprise visit from becoming an unannounced overnight stay?",
    ],
}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def re_stem(text: str) -> str:
    out = []
    for tok in text.lower().split():
        if any(ch.isdigit() for ch in tok):
            continue
        if "." in tok and not tok.endswith((".", "?", ",")):
            continue
        out.append(tok.strip(".,:;?'\""))
    return " ".join(out)


def _load_prompts(path: Path):
    return json.loads(path.read_text()).get("prompts") or []


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
        sources[str(path)] = {
            "n_protected_texts": len(protected)}
    return exist_ids, exist_texts, protected, sources


def main():
    dev3_sha = hashlib.sha256(DEV3.read_bytes()).hexdigest()
    assert dev3_sha == DEV3_SHA, (dev3_sha, DEV3_SHA)
    dev3 = json.loads(DEV3.read_text())
    assert dev3["splits"]["train"] == []
    assert len(dev3["splits"]["val"]) == 70

    exist_ids, exist_texts, protected, sources = collect_blocked()
    train_prompts, train_ids = [], []
    id_hits, text_hits, prot_hits = [], [], []
    for dom, texts in P.items():
        assert len(texts) == 43, (dom, len(texts))
        assert len(set(texts)) == 43, (dom, "internal dup")
        stems = [re_stem(t) for t in texts]
        assert len(set(stems)) == 43, (dom, "stem collapse")
        for i, text in enumerate(texts):
            cat = f"{dom}_{137 + i:03d}"
            if cat in exist_ids:
                id_hits.append(cat)
            n = _norm(text)
            if n in exist_texts:
                text_hits.append({"category": cat, "prompt": text})
            if n in protected:
                prot_hits.append({"category": cat, "prompt": text})
            train_prompts.append({"category": cat, "prompt": text,
                                  "domain": dom})
            train_ids.append(cat)

    assert not id_hits, f"ID overlap: {id_hits}"
    assert not text_hits, f"text overlap: {text_hits}"
    assert not prot_hits, f"protected overlap: {prot_hits}"
    assert len(train_ids) == 301
    assert len({_norm(p["prompt"]) for p in train_prompts}) == 301
    assert len(set(train_ids)) == 301

    # Combined capture manifest: val first (phase order).
    val_prompts = list(dev3["prompts"])
    val_ids = list(dev3["splits"]["val"])
    prompts = val_prompts + train_prompts
    body = {
        "classification": "STAGE1_SCALE1000_CORPUS",
        "purpose": "SCALE-DEV3 holdout + 301 SCALE1000 train "
                   "additions (1001 = 7x143 with frozen 700)",
        "splits": {"train": train_ids, "val": val_ids},
        "prompts": prompts,
        "frozen_dev3_sha256": DEV3_SHA,
        "author_note": "IDs 137-179 train; DEV3 127-136 val "
                       "unchanged; 43 distinct tasks/domain",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=1) + "\n")
    man_sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    src_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit = {
        "classification": "SCALE1000_TRAIN_OVERLAP_AUDIT",
        "n_train": 301,
        "n_val": 70,
        "train_ids": train_ids,
        "blocked_sources": sources,
        "id_overlap": id_hits,
        "text_overlap": text_hits,
        "protected_overlap": prot_hits,
        "dev3_sha256_checked": dev3_sha,
        "all_empty_required": True,
        "manifest_sha256": man_sha,
        "author_sha256": src_sha,
    }
    AUDIT.write_text(json.dumps(audit, indent=1) + "\n")
    print(f"wrote {OUT}: {len(train_ids)} train + {len(val_ids)} val")
    print("manifest sha256:", man_sha)
    print("author sha256:", src_sha)
    print("audit sha256:", hashlib.sha256(
        AUDIT.read_bytes()).hexdigest())
    print("DEV3 sha unchanged:", dev3_sha)
    print("overlap checks empty:",
          not id_hits and not text_hits and not prot_hits)


if __name__ == "__main__":
    main()
