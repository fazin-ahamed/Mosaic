"""Author the SCALE700 + SCALE-DEV2 corpus (roadmap S9, S11).

560 new prompts, 80 per domain:
  dev2   {dom}_47..{dom}_56  (10/domain, 70)  -> SCALE-DEV2
  train  {dom}_57..{dom}_126 (70/domain, 490) -> SCALE700 additions

Guards: text-disjoint from all 322 existing prompts (126
original + 196 scale) and G1/G2 protected texts; within-set
uniqueness; deterministic seed. Manifest: val=dev2 first,
then train (phase-ordered for the capture engine).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

OUT = Path("artifacts/d3_scale700/prompts_scale700.json")
EXISTING = [Path("artifacts/stage1_prompts.json"),
            Path("artifacts/d3_scale/prompts_scale.json")]

rng = random.Random(700210)

# ---------------- math: parameterized families ----------------
def math_prompts():
    out = []
    while len(out) < 80:
        fam = len(out) % 10
        if fam == 0:
            a, b = rng.randint(12, 99), rng.randint(12, 99)
            t = f"What is {a} times {b}?"
        elif fam == 1:
            b = rng.randint(3, 19); q = rng.randint(2, 15)
            t = f"What is {b*q} divided by {b}?"
        elif fam == 2:
            p = rng.choice([5, 10, 12, 15, 20, 25, 30, 35,
                            40, 45, 60, 75])
            n = rng.choice([60, 80, 120, 150, 180, 240,
                            320, 360, 450, 480])
            t = f"What is {p} percent of {n}?"
        elif fam == 3:
            a = rng.randint(3, 9); x = rng.randint(3, 14)
            b = rng.randint(2, 40)
            t = f"Solve for x: {a}x + {b} = {a*x+b}."
        elif fam == 4:
            h = rng.choice([2, 3, 4, 5, 6])
            s = rng.choice([40, 60, 75, 80, 90, 120, 150])
            t = (f"A train travels {h*s} kilometers in {h} "
                 f"hours. What is its average speed?")
        elif fam == 5:
            p = rng.choice([40, 50, 60, 70, 80, 90, 120])
            d = rng.choice([10, 15, 20, 25, 30, 40])
            t = (f"A jacket costs {p} dollars and is "
                 f"discounted by {d} percent. What is the "
                 f"sale price?")
        elif fam == 6:
            l = rng.randint(6, 25); w = rng.randint(4, 18)
            t = (f"What is the area of a rectangle with "
                 f"length {l} and width {w}?")
        elif fam == 7:
            a, r = rng.randint(2, 9), rng.randint(2, 5)
            s = [a * r**i for i in range(3)]
            t = (f"What is the next number in the sequence "
                 f"{s[0]}, {s[1]}, {s[2]}?")
        elif fam == 8:
            k = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            12, 15, 20, 25])
            pair = rng.choice([
                ("kilometers", "meters"),
                ("meters", "centimeters"),
                ("kilograms", "grams"),
                ("liters", "milliliters"),
                ("hours", "minutes"),
                ("minutes", "seconds"),
                ("megabytes", "kilobytes")])
            t = (f"Convert {k} {pair[0]} to "
                 f"{pair[1]}.")
        else:
            n = rng.randint(3, 9)
            c = rng.choice([(1,2), (3,4), (2,3), (1,3),
                            (2,5), (3,5), (5,8)])
            t = (f"A recipe needs {c[0]} over {c[1]} cup of "
                 f"sugar per batch. How much for {n} "
                 f"batches?")
        if t not in out:
            out.append(t)
    return out

# ---------------- json: entity x fields ----------------
def json_prompts():
    ents = ["monitor", "keyboard", "printer", "tablet",
            "camera", "speaker", "headphones", "smartwatch",
            "television", "bicycle", "truck", "boat",
            "house", "apartment", "office", "hotel",
            "cafe", "library", "museum", "park", "school",
            "clinic", "pharmacy", "bank", "store", "gym",
            "stadium", "theater", "cinema", "farm",
            "factory", "airport", "bridge", "tower",
            "garden", "bakery", "bookstore", "barbershop",
            "workshop", "warehouse", "observatory",
            "greenhouse", "aquarium", "windmill", "lighthouse",
            "campground", "marketplace", "monastery",
            "vineyard", "orchard", "hatchery", "apiary",
            "planetarium", "amphitheater", "boardwalk",
            "marina", "observatory deck", "reservoir",
            "quarry", "foundry", "mill", "kiln",
            "forge", "brewery", "distillery", "roastery",
            "creamery", "smokery", "cannery", "bindery",
            "print shop", "dye house", "loom house",
            "glasshouse", "boathouse", "gatehouse",
            "clubhouse", "firehouse", "courthouse",
            "clocktower", "watermill", "granary", "silo"]
    assert len(ents) >= 80
    fieldsets = [
        "brand, model, price, and warranty years",
        "name, location, capacity, and rating",
        "id, title, category, and tags array",
        "owner, address, size, and features array",
        "type, color, weight, and dimensions object",
    ]
    verbs = ["Output JSON for", "Create JSON for",
             "Return JSON for", "Generate JSON for"]
    # dev2: entities 0-9, distinct (fieldset, verb) per prompt
    dev2 = [f"{verbs[i % 4]} a {ents[i]} with "
            f"{fieldsets[i % 5]}." for i in range(10)]
    # train: entities 10-79, template spread
    train = []
    for i in range(70):
        e = ents[10 + i]
        f = fieldsets[(i * 3 + 1) % 5]
        v = verbs[(i * 2 + 1) % 4]
        train.append(f"{v} a {e} with {f}.")
    assert len(set(dev2 + train)) == 80
    return dev2 + train

# ---------------- retrieval: doc x angle ----------------
def retrieval_prompts():
    docs = ["quarterly report", "annual review",
            "user manual", "safety handbook",
            "meeting notes", "project charter",
            "policy document", "vendor proposal",
            "compliance audit", "research summary",
            "incident report", "weather advisory",
            "course syllabus", "product warranty",
            "service contract", "release changelog",
            "inventory log", "onboarding guide",
            "maintenance schedule", "budget summary",
            "press release", "grant application",
            "inspection certificate", "evacuation plan",
            "style guide", "data retention policy",
            "recruitment brief", "lease agreement"]
    angles = [
        "what was the {k} in the latest update",
        "who is responsible for the pilot phase",
        "when is the submission deadline",
        "how much funding was approved",
        "which team exceeded its targets",
        "why was the timeline extended",
        "how is the escalation procedure described",
        "what eligibility rules are listed",
    ]
    keys = ["revenue growth", "attrition rate",
            "error rate", "throughput limit"]
    # dev2: docs 0-9, distinct angle per prompt
    dev2 = []
    for i in range(10):
        d = docs[i]
        a = angles[i % 8]
        k = keys[i % 4]
        dev2.append(
            f"According to the {d}, {a.format(k=k)}?")
    # train: docs 10-27 (disjoint) x angles with spread
    train = []
    i = 0
    while len(train) < 70:
        d = docs[10 + (i % 18)]
        a = angles[(i // 18 + i % 5) % 8]
        k = keys[(i // 3) % 4]
        t = f"According to the {d}, {a.format(k=k)}?"
        if t not in train and t not in dev2:
            train.append(t)
        i += 1
    assert len(set(dev2 + train)) == 80
    return dev2 + train

# ---------------- agent: task family x object ----------------
def agent_prompts():
    objs = ["records.csv", "archive.tar.gz", "notes.md",
            "config.yaml", "backup.db", "index.html",
            "output.log", "server.pid", "data.json",
            "report.pdf", "metrics.txt", "queue.tmp",
            "session.bin", "manifest.xml"]
    fams = [
        "Show the first 20 lines of {o}.",
        "Count the lines in {o} and print the total.",
        "Compress {o} into a zip archive.",
        "Display the file type and size of {o}.",
        "Make {o} read-only for everyone.",
        "Find all references to error inside {o}.",
        "Move {o} into a folder called processed.",
        "Show the last modified timestamp of {o}.",
        "Create a backup copy of {o} with a dated suffix.",
        "Print the first and last line of {o}.",
        "Check whether {o} contains any empty lines.",
        "Convert {o} to uppercase and save the result.",
        "List the permissions of {o} in octal form.",
        "Append the current date to the end of {o}.",
        "Verify that {o} is a valid file and not a symlink.",
        "Show the number of words in {o}.",
    ]
    # dev2: fams 0-9, one distinct object each (distinct tasks)
    dev2 = [fams[i].format(o=objs[i]) for i in range(10)]
    # train: fams 10-15 x all objs, then fams 0-9 x objs 4-13
    train = []
    for i in range(6):
        for j in range(14):
            train.append(
                fams[10 + i].format(o=objs[j]))
    for i in range(10):
        for j in range(4, 14):
            train.append(
                fams[i].format(o=objs[j]))
    # dedupe and trim to 70 with spread
    seen, out = set(), []
    for t in train:
        if t not in seen and t not in dev2:
            seen.add(t); out.append(t)
    assert len(out) >= 70, len(out)
    out = out[:70]
    assert len(set(dev2 + out)) == 80
    return dev2 + out

# ---------------- coding: 80 distinct tasks ----------------
def coding_prompts():
    return [
        "Write a function that capitalizes the first letter of every word in a sentence.",
        "Write a function that finds the second smallest number in a list.",
        "Write a function that checks if two strings are anagrams.",
        "Write a function that converts a temperature from Celsius to Fahrenheit.",
        "Write a function that returns the most frequent element in a list.",
        "Write a function that sums only the even numbers in a list.",
        "Write a function that rotates a list by k positions.",
        "Write a function that checks if a number is prime.",
        "Write a function that formats a duration in seconds as hours, minutes and seconds.",
        "Write a function that finds all indices of a value in a list.",
        "Write a function that merges two sorted lists into one sorted list.",
        "Write a function that removes all punctuation from a string.",
        "Write a function that computes the average of a list ignoring outliers.",
        "Write a function that checks if a year is a century year.",
        "Write a function that generates the first n square numbers.",
        "Write a function that counts the vowels at even positions in a string.",
        "Write a function that replaces spaces with underscores in a filename.",
        "Write a function that checks if a list is sorted.",
        "Write a function that interleaves two lists of equal length.",
        "Write a function that returns the shortest string in a list.",
        "Write a function that computes the dot product of two vectors.",
        "Write a function that finds the longest word in a sentence.",
        "Write a function that converts a decimal number to binary.",
        "Write a function that counts occurrences of each character in a string.",
        "Write a function that removes duplicate adjacent characters from a string.",
        "Write a function that calculates compound interest over n years.",
        "Write a function that checks if a string contains balanced parentheses.",
        "Write a function that returns the union of two lists without duplicates.",
        "Write a function that finds the difference between two lists.",
        "Write a function that truncates a string to a maximum length with an ellipsis.",
        "Write a function that generates a random password of a given length.",
        "Write a function that checks if a sentence is a pangram.",
        "Write a function that computes the median of a list of numbers.",
        "Write a function that reverses the order of words in a sentence.",
        "Write a function that finds the longest run of identical characters.",
        "Write a function that converts a list of tuples into a dictionary.",
        "Write a function that rounds a price to the nearest five cents.",
        "Write a function that checks if two rectangles overlap.",
        "Write a function that returns the day of the year for a given date.",
        "Write a function that normalizes a list of numbers to the range zero to one.",
        "Write a function that counts the number of syllables in a word.",
        "Write a function that finds all permutations of a three-element list.",
        "Write a function that checks if a number is a perfect square.",
        "Write a function that removes every nth element from a list.",
        "Write a function that finds the most common word in a paragraph.",
        "Write a function that converts hours and minutes into total minutes.",
        "Write a function that checks if a string follows a valid date format.",
        "Write a function that computes the sum of digits of an integer.",
        "Write a function that swaps the case of every letter in a string.",
        "Write a function that finds the intersection of three sets.",
        "Write a function that pads a number with leading zeros to a fixed width.",
        "Write a function that checks if a list contains a palindrome.",
        "Write a function that calculates the distance between two points.",
        "Write a function that groups a list of words by their first letter.",
        "Write a function that returns the nth row of a multiplication table.",
        "Write a function that checks if a word appears twice in a sentence.",
        "Write a function that converts seconds into a readable uptime string.",
        "Write a function that finds the first non-repeating character in a string.",
        "Write a function that builds an acronym from a phrase.",
        "Write a function that determines the season from a month number.",
        "Write a function that counts how many numbers in a list are above a threshold.",
        "Write a function that replaces every occurrence of a substring with another.",
        "Write a function that computes the factorial of a number iteratively.",
        "Write a function that finds the key with the maximum value in a dictionary.",
        "Write a function that checks whether all elements in a list are unique.",
        "Write a function that converts a snake_case identifier to camelCase.",
        "Write a function that returns the elements at even indices of a list.",
        "Write a function that estimates the reading time of a text in minutes.",
        "Write a function that validates a credit card number length.",
        "Write a function that finds the smallest missing positive integer in a list.",
        "Write a function that repeats a string n times with a separator.",
        "Write a function that computes the angle between clock hands at a given time.",
        "Write a function that extracts all numbers from a string of mixed text.",
        "Write a function that checks if one string is a rotation of another.",
        "Write a function that computes the body mass index from weight and height.",
        "Write a function that returns a list of running totals from a list of numbers.",
        "Write a function that masks all but the last four digits of a number sequence.",
        "Write a function that finds the longest increasing run in a list.",
        "Write a function that formats a name as surname, first name.",
        "Write a function that checks if a triangle is valid from three side lengths.",
        "Write a function that converts a list of Celsius readings to a daily summary.",
    ]

# ---------------- longform: topics ----------------
def longform_prompts():
    return [
        "Explain how a solar panel converts sunlight into electricity.",
        "Describe how blood circulates through the human heart.",
        "Explain how a bicycle brake stops the wheel.",
        "Describe how vaccines prevented smallpox.",
        "Explain how a microwave heats food.",
        "Describe how rain forms in clouds.",
        "Explain how an elevator moves between floors safely.",
        "Describe how bees make honey from nectar.",
        "Explain how a zipper opens and closes.",
        "Describe how the immune system fights a cold.",
        "Explain how a dam generates hydroelectric power.",
        "Describe how bread rises in the oven.",
        "Explain how a refrigerator compressor works.",
        "Describe how volcanoes erupt.",
        "Explain how a toilet flush mechanism operates.",
        "Describe how photosynthesis produces oxygen.",
        "Explain how a mechanical watch keeps time.",
        "Describe how tides follow the moon.",
        "Explain how a washing machine spins clothes dry.",
        "Describe how a seed grows into a tree.",
        "Explain how sound travels through air.",
        "Describe how a rocket escapes gravity.",
        "Explain how a thermostat maintains temperature.",
        "Describe how spiders spin webs.",
        "Explain how a camera aperture controls light.",
        "Describe how glaciers carve valleys.",
        "Explain how an air conditioner cools a room.",
        "Describe how the digestive system breaks down food.",
        "Explain how a lighthouse beam is visible far away.",
        "Describe how ant colonies find food.",
        "Explain how a lock and key mechanism works.",
        "Describe how lightning strikes form.",
        "Explain how a steam engine converted heat to motion.",
        "Describe how kidneys filter blood.",
        "Explain how a parachute slows a fall.",
        "Describe how coral polyps build reefs.",
        "Explain how a hydraulic press multiplies force.",
        "Describe how migratory birds navigate.",
        "Explain how a battery stores chemical energy.",
        "Describe how waves erode a coastline.",
        "Explain how a printing press mass-produces pages.",
        "Describe how muscles contract during exercise.",
        "Explain how a windmill pumps water.",
        "Describe how fog forms on a cold morning.",
        "Explain how a subway signaling system prevents collisions.",
        "Describe how trees transport water upward.",
        "Explain how a smoke detector senses fire.",
        "Describe how penguins survive Antarctic winters.",
        "Explain how a sailboat moves against the wind.",
        "Describe how cheese is cultured from milk.",
        "Explain how an escalator keeps its steps level.",
        "Describe how earthquakes reshape coastlines.",
        "Explain how a hearing aid amplifies sound.",
        "Describe how cacti conserve water in deserts.",
        "Explain how a fire extinguisher stops combustion.",
        "Describe how the eye focuses on near and far objects.",
        "Explain how a water tower supplies a town.",
        "Describe how salmon find their spawning stream.",
        "Explain how a helicopter rotor generates lift.",
        "Describe how yogurt forms from milk cultures.",
        "Explain how a compass needle finds north.",
        "Describe how avalanches build up and release.",
        "Explain how a sewing machine stitches fabric.",
        "Describe how moths navigate by moonlight.",
        "Explain how a sump pump keeps a basement dry.",
        "Describe how dunes shift across a desert.",
        "Explain how a guitar string produces different pitches.",
        "Describe how owls hunt in total darkness.",
        "Explain how a diving bell works underwater.",
        "Describe how frost patterns form on windows.",
        "Explain how a Ferris wheel stays balanced.",
        "Describe how termites build mounds with ventilation.",
        "Explain how a metronome keeps a steady beat.",
        "Describe how hot springs heat themselves.",
        "Explain how a turntable reads grooves into music.",
        "Describe how octopuses change their skin color.",
        "Explain how a grain silo protects its contents.",
        "Describe how fireflies synchronize their flashes.",
        "Explain how a canal lock raises and lowers boats.",
        "Describe how spiders rebuild damaged webs.",
    ]

# ---------------- convo: situations ----------------
def convo_prompts():
    return [
        "How do I choose between buying and renting a home?",
        "What should I say when I disagree with my boss in a meeting?",
        "How can I make my commute less exhausting?",
        "What are good questions to ask at a job interview?",
        "How do I help a friend who is stressed about exams?",
        "What should I pack for a week-long beach vacation?",
        "How do I start a conversation at a wedding where I know nobody?",
        "What is a considerate way to ask a neighbor to quiet their dog?",
        "How can I keep houseplants alive when I travel often?",
        "What should I write in a card for a retiring colleague?",
        "How do I decide how much to spend on a wedding gift?",
        "What are some ways to enjoy a weekend without spending money?",
        "How should I react when someone takes credit for my idea?",
        "What can I do to make mornings smoother with kids?",
        "How do I ask a friend to repay a loan without damaging the friendship?",
        "What are tips for taking care of a new pair of shoes?",
        "How can I be more patient in traffic?",
        "What is the best way to learn the rules of a new sport?",
        "How do I choose a book I will actually finish?",
        "What should I consider before adopting a second cat?",
        "How can I support a friend running their first marathon?",
        "What is a polite way to leave a party early?",
        "How do I handle receiving a gift I do not like?",
        "What are some ideas for a memorable anniversary on a budget?",
        "How can I keep in touch with friends in other time zones?",
        "What should I do when I forget someone's name mid-conversation?",
        "How do I set up a workspace that helps me focus?",
        "What is a good way to break a long phone habit?",
        "How should I prepare for hosting dinner guests with dietary restrictions?",
        "What can I say to comfort someone after a job loss?",
        "How do I politely decline to be a bridesmaid?",
        "What are strategies for splitting bills fairly among roommates?",
        "How can I encourage a shy child to join group activities?",
        "What should I pack for a first camping trip?",
        "How do I stay calm when giving a presentation?",
        "What is a thoughtful way to celebrate a coworker's promotion?",
        "How can I make a small apartment feel bigger?",
        "What should I say to a friend moving abroad?",
        "How do I choose a reliable used car seller?",
        "What are good habits before going to sleep?",
        "How can I make video calls less tiring?",
        "What is the best way to store winter clothes?",
        "How should I handle a dispute with a neighbor about parking?",
        "What can I do to be a better listener in arguments?",
        "How do I plan a surprise party without being caught?",
        "What are some ways to practice a language daily?",
        "How should I respond to a vague apology?",
        "What is a fair way to divide household chores with a partner?",
        "How can I keep my phone battery healthy for years?",
        "What should I bring to a potluck I cannot cook for?",
        "How do I choose a podcast worth my commute time?",
        "What is a kind way to tell a friend their advice did not help?",
        "How can I prepare my home for a new puppy?",
        "What should I do when two friends expect me to take sides?",
        "How do I make time for exercise with a busy schedule?",
        "What are tips for photographing a child's first steps?",
        "How should I greet new neighbors who just moved in?",
        "What can I say at a funeral when I knew the person briefly?",
        "How do I decide whether to repair or replace an appliance?",
        "What is a good routine for watering plants while away?",
        "How can I politely ask houseguests to tidy up?",
        "What should I write in a first message to an old friend?",
        "How do I choose a hobby I will stick with?",
        "What are ways to celebrate small wins at work?",
        "How can I help my parents use their new phone?",
        "What is the best way to back up family photos?",
        "How should I handle a coworker who talks over me?",
        "What can I do to enjoy cooking for one person?",
        "How do I pick a movie everyone in the family will like?",
        "What should I say when asked about my salary expectations?",
        "How can I make a hospital visit more comforting?",
        "What is a sensible way to handle junk mail?",
        "How do I prepare for my first long-haul flight?",
        "What are good icebreakers for a team that just formed?",
        "How should I thank a mentor at the end of a program?",
        "What can I do to make laundry day less of a chore?",
        "How do I choose between two apartments I like equally?",
        "What is a respectful way to change the subject?",
        "How can I keep my bicycle from being stolen in the city?",
        "How can I keep my workspace tidy without obsessing?",
    ]

GEN = {"math": math_prompts, "json": json_prompts,
       "retrieval": retrieval_prompts,
       "agent": agent_prompts, "coding": coding_prompts,
       "longform": longform_prompts, "convo": convo_prompts}


def main():
    exist_texts = set()
    exist_ids = set()
    for p in EXISTING:
        d = json.loads(p.read_text())
        for pp in d["prompts"]:
            exist_texts.add(" ".join(pp["prompt"].lower()
                                     .split()))
            exist_ids.add(pp["category"])
    protected = set()
    for name in ("routes_overlap_corpus_3060.json",
                 "g2_cert_prompts.json",
                 "g2_calibration_prompts.json"):
        for entry in json.loads(
                (Path("artifacts") / name).read_text()
        )["prompts"]:
            protected.add(" ".join(
                entry["prompt"].lower().split()))

    train, val, prompts = [], [], []
    for dom, gen in GEN.items():
        texts = gen()
        assert len(texts) >= 80 and len(set(texts)) >= 80, (
            dom, len(texts), len(set(texts)))
        texts = texts[:80]
        for t in texts:
            norm = " ".join(t.lower().split())
            assert norm not in exist_texts, f"dup: {t}"
            assert norm not in protected, f"protected: {t}"
        for i, t in enumerate(texts):
            num = 47 + i                     # 47..126
            cat = f"{dom}_{num:03d}"
            assert cat not in exist_ids
            prompts.append({"category": cat, "prompt": t})
            (val if i < 10 else train).append(cat)

    assert len(val) == 70 and len(train) == 490
    norms = [" ".join(p["prompt"].lower().split())
             for p in prompts]
    assert len(set(norms)) == 560

    # generators return dev2-first per domain; build the
    # manifest with the val block first (phase order)
    val_cats = set(val)
    prompts = ([p for p in prompts
                if p["category"] in val_cats]
               + [p for p in prompts
                  if p["category"] not in val_cats])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"classification": "STAGE1_SCALE700_CORPUS",
         "purpose": "SCALE-DEV2 holdout + SCALE700 train "
                    "additions",
         "splits": {"train": train, "val": val},
         "prompts": prompts}, indent=1))
    print(f"wrote {OUT}: {len(train)} train + {len(val)} val")
    print("sha256:", hashlib.sha256(
        OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
