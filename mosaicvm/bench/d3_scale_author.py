"""Author the SCALE210 + SCALE-DEV70 corpus (roadmap S13-S15).

196 new prompts, 28 per domain:
  train  {dom}_19..{dom}_36  (18/domain, 126) -> SCALE210
  val    {dom}_37..{dom}_46  (10/domain, 70)  -> SCALE-DEV70

Guards: text-disjoint from all 126 existing prompts and from
G1/G2 protected prompts; manifest structure matches
stage1_prompts.json (splits/prompts) for stage1_record.py.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("artifacts/d3_scale/prompts_scale.json")
EXISTING = Path("artifacts/stage1_prompts.json")

P = {
"coding": [
 "Write a function that checks if a string is a palindrome ignoring punctuation.",
 "How do I merge two dictionaries in Python without losing keys?",
 "Write a SQL query that finds the second-highest salary in an employees table.",
 "Explain what a race condition is and give a small example.",
 "Write a function that returns the nth Fibonacci number using iteration.",
 "How do I split a string into words while stripping punctuation?",
 "Write a regular expression that matches a valid IPv4 address.",
 "Explain the difference between shallow copy and deep copy.",
 "Write a function that finds the longest common prefix of a list of strings.",
 "How do I read a CSV file line by line in Python without loading it all into memory?",
 "Write a bash script that renames all .txt files in a folder to .md.",
 "Explain what Big-O notation measures with two examples.",
 "Write a function that counts word frequencies in a sentence.",
 "How do I safely parse an integer from user input in Python?",
 "Write a class representing a bank account with deposit and withdraw methods.",
 "Explain the difference between HTTP GET and POST requests.",
 "Write a function that removes all vowels from a string.",
 "How do I sort a list of dictionaries by a specific key in Python?",
 "Write a function that validates an email address format.",
 "Explain what a hash collision is and how it is usually handled.",
 "Write a function that computes the Greatest Common Divisor recursively.",
 "How do I use a context manager to open two files at once in Python?",
 "Write a function that finds the first duplicate element in a list.",
 "Explain the difference between compiled and interpreted languages.",
 "Write a function that checks whether a year is a leap year.",
 "How do I reverse the key-value pairs of a dictionary in Python?",
 "Write a small function that shuffles a list without using the built-in shuffle.",
 "Explain what dependency injection is with a one-sentence example.",
],
"math": [
 "What is 23 times 17?",
 "A train travels 240 kilometers in 3 hours. What is its average speed?",
 "Solve for x: 4x + 12 = 48.",
 "What is 15 percent of 320?",
 "A shirt costs 45 dollars and is discounted by 20 percent. What is the sale price?",
 "Simplify the fraction 84 over 126.",
 "What is the area of a circle with radius 7? Use pi approximately 3.14.",
 "A car uses 8 liters of fuel per 100 kilometers. How much fuel for 350 kilometers?",
 "Solve for y: 3y - 7 = 2y + 5.",
 "What is the square root of 784?",
 "A classroom has 30 students and 60 percent are girls. How many boys are there?",
 "What is 2 to the power of 12?",
 "A recipe calls for 2 and a half cups of flour for 4 servings. How much for 10 servings?",
 "What is the perimeter of a rectangle with length 15 and width 8?",
 "Solve for z: z divided by 6 equals 14.",
 "What is 9 factorial divided by 7 factorial?",
 "A phone battery drops from 80 percent to 20 percent in 5 hours. What is the drain rate per hour?",
 "What is the next number in the sequence 2, 6, 18, 54?",
 "Convert 3.75 kilometers to meters.",
 "What is the volume of a cube with edge length 4?",
 "Solve for x: 7x minus 21 equals 0.",
 "A worker earns 18 dollars per hour and works 37 hours. What is the gross pay?",
 "What is 0.625 as a fraction in simplest form?",
 "A tank holds 500 liters and is three quarters full. How many liters are in it?",
 "What is the sum of the interior angles of a triangle in degrees?",
 "If 5 notebooks cost 12 dollars, what do 8 notebooks cost at the same rate?",
 "What is 121 divided by 11 plus 13?",
 "A ladder leans against a wall forming a right triangle with legs 9 and 12. How long is the ladder?",
],
"agent": [
 "List all files in the current directory sorted by size.",
 "Show me the first 15 lines of a file named config.yaml.",
 "Find all files modified in the last 24 hours.",
 "Create an empty file called notes.txt and then show its permissions.",
 "Show me how much free disk space is available.",
 "Kill the process listening on port 3000.",
 "Copy a file called report.pdf to a backup folder.",
 "Show the current working directory and then change to the home directory.",
 "Search for the word error inside all log files in this folder.",
 "Display the last 10 commands you executed.",
 "Create a compressed archive of the documents folder.",
 "Show me the environment variables containing the word PATH.",
 "Rename all files with extension .jpeg to .jpg in the current folder.",
 "Print the number of lines, words, and characters in readme.md.",
 "Show the date from 30 days ago in year-month-day format.",
 "Download a file from a URL and save it as latest.tar.gz.",
 "List all running processes sorted by memory usage.",
 "Create a new user group called developers.",
 "Show the differences between two files called old.txt and new.txt.",
 "Display the contents of a file with line numbers.",
 "Find and delete all empty directories under the current folder.",
 "Show me the current system uptime and load average.",
 "Set the file permissions of script.sh to executable for the owner only.",
 "Show the largest 5 files anywhere under /var/log.",
 "Schedule a task to run a cleanup script every day at 3 AM.",
 "Display the last 5 login events on this machine.",
 "Find all symbolic links in the current directory tree.",
 "Print a recursive listing of the src folder but exclude hidden files.",
],
"convo": [
 "What should I pack for a weekend trip to the mountains?",
 "How do I tell my roommate their music is too loud without offending them?",
 "What are some thoughtful gift ideas for a friend who loves cooking?",
 "How can I stay focused while working from home?",
 "What is a good way to end a phone call politely when you are busy?",
 "How do I ask my manager for feedback without seeming insecure?",
 "What are some fun indoor activities for a rainy Sunday?",
 "How should I respond when someone gives me a compliment?",
 "What is the best way to break the ice at a networking event?",
 "How can I help a friend who is moving to a new city?",
 "What are some strategies for saving money on groceries?",
 "How do I decide between two job offers?",
 "What should I say to a colleague who just lost a family member?",
 "How can I make my morning routine less stressful?",
 "What are some ways to celebrate a birthday on a small budget?",
 "How do I politely ask guests to leave after a dinner party?",
 "What is a good way to learn someone's name and actually remember it?",
 "How can I support a partner who had a rough day at work?",
 "What are some conversation starters for a long car ride?",
 "How do I set boundaries with a friend who calls too late at night?",
 "What should I consider before adopting a rescue dog?",
 "How can I make new friends after moving to a new town?",
 "What is a kind way to tell someone they have food in their teeth?",
 "How do I prepare for a first dinner with my partner's parents?",
 "What should I keep in mind when shopping for a used bicycle?",
 "How can I wind down effectively at the end of a stressful day?",
 "What should I write in a thank-you card for a teacher?",
 "How do I gracefully handle being double-booked for two events?",
],
"retrieval": [
 "According to the quarterly report, which region had the highest sales growth?",
 "What does the user manual say about resetting the device to factory settings?",
 "According to the meeting notes, who was assigned to lead the design review?",
 "What budget figure did the finance summary approve for the marketing campaign?",
 "According to the email thread, when is the client presentation scheduled?",
 "What does the policy document state about remote work eligibility?",
 "According to the audit, which two departments exceeded their travel budgets?",
 "What safety procedure does the training handbook describe for chemical spills?",
 "According to the project plan, what are the three main deliverables for phase two?",
 "What did the customer feedback survey identify as the top complaint?",
 "According to the release notes, which bugs were fixed in version 2.4?",
 "What does the contract specify as the notice period for termination?",
 "According to the inventory log, how many units of item 4021 remain in stock?",
 "What does the research summary conclude about the new battery chemistry?",
 "According to the onboarding guide, which forms must new hires submit in week one?",
 "What attendance threshold does the course syllabus set for passing?",
 "According to the incident report, what time was the outage first detected?",
 "What does the vendor proposal include that the competitor's does not?",
 "According to the committee charter, how often are elections held?",
 "What staffing change did the organization chart show for the support team?",
 "According to the recipe, at what temperature should the oven be preheated?",
 "What does the warranty cover according to the product documentation?",
 "According to the weather advisory, which roads are closed due to flooding?",
 "What login issue does the IT bulletin describe and how is it resolved?",
 "According to the annual review, what was the employee retention rate?",
 "What does the leasing agreement say about pets in the building?",
 "According to the changelog, which feature was deprecated last quarter?",
 "What are the eligibility criteria listed in the scholarship announcement?",
],
"longform": [
 "Explain how a refrigerator keeps food cold.",
 "Describe how an earthquake is measured and classified.",
 "Explain how vaccines train the immune system.",
 "Describe the life cycle of a star from formation to collapse.",
 "Explain how GPS determines your location.",
 "Describe how a jury trial proceeds from selection to verdict.",
 "Explain how wifi transmits data through the air.",
 "Describe the process of refining crude oil into gasoline.",
 "Explain how credit scores are calculated and used.",
 "Describe how a bee colony organizes its labor.",
 "Explain why the sky is blue during the day and red at sunset.",
 "Describe how a bill becomes law in a national legislature.",
 "Explain how noise-cancelling headphones work.",
 "Describe how a package travels from an online order to your front door.",
 "Explain how databases index data for fast lookups.",
 "Describe how coral reefs form over centuries.",
 "Explain what causes inflation in an economy.",
 "Describe how a wind turbine converts wind into electricity.",
 "Explain how sleep cycles progress through the night.",
 "Describe the process of chocolate from cacao bean to bar.",
 "Explain how search engines rank web pages.",
 "Describe how antibodies recognize viruses.",
 "Explain how a submarine controls its depth.",
 "Describe the migration patterns of arctic terns.",
 "Explain how fiber optic cables carry information.",
 "Describe how an airport handles an arriving international flight.",
 "Explain why leaves change color in autumn.",
 "Describe how a library catalogues and retrieves books.",
],
"json": [
 "Output JSON for a laptop with brand, model, price, and specs object.",
 "Create JSON for a restaurant with name, cuisine, rating, and menu array.",
 "Return JSON for a student with id, name, major, and enrolled courses array.",
 "Output JSON for a playlist with title, owner, and songs array of objects.",
 "Create JSON for an apartment with rooms count, area, rent, and amenities array.",
 "Return JSON for a flight with airline, number, departure, arrival, and stops.",
 "Output JSON for a movie with title, director, year, runtime, and cast array.",
 "Create JSON for an order with order id, customer, items array, and total.",
 "Return JSON for a bookshelf with location and books array of titles.",
 "Output JSON for a gym with name, opening hours, and equipment list.",
 "Create JSON for a hospital with departments array and bed capacity.",
 "Return JSON for a recipe with name, ingredients array, and step count.",
 "Output JSON for a car with make, model, year, color, and mileage.",
 "Create JSON for a conference with name, dates, venue, and speakers array.",
 "Return JSON for an inventory item with sku, name, quantity, and warehouse.",
 "Output JSON for a team with name, sport, founded year, and roster array.",
 "Create JSON for a subscription with plan, price, renewal date, and features array.",
 "Return JSON for a weather report with city, temperature, humidity, and forecast array.",
 "Output JSON for a job posting with title, company, salary range, and requirements array.",
 "Create JSON for a course with code, title, credits, and prerequisites array.",
 "Return JSON for a plant with species, height, water needs, and sunlight.",
 "Output JSON for a bank transaction with id, amount, timestamp, and status.",
 "Create JSON for a museum with name, city, exhibits array, and ticket price.",
 "Return JSON for a podcast with title, host, episode count, and topics array.",
 "Output JSON for a sensor reading with device id, timestamp, value, and unit.",
 "Create JSON for a delivery with tracking number, status, and history array.",
 "Return JSON for a hotel with name, stars, room types array, and address.",
 "Output JSON for a review with author, rating, date, and comment text.",
],
}

def main():
    existing = json.loads(EXISTING.read_text())
    exist_texts = {" ".join(p["prompt"].lower().split())
                   for p in existing["prompts"]}
    exist_ids = {p["category"] for p in existing["prompts"]}
    # G1/G2 protected texts
    protected = set()
    for name in ("routes_overlap_corpus_3060.json",
                 "g2_cert_prompts.json",
                 "g2_calibration_prompts.json"):
        for entry in json.loads(
                (Path("artifacts") / name).read_text())["prompts"]:
            protected.add(" ".join(
                entry["prompt"].lower().split()))

    train, val, prompts = [], [], []
    for dom, texts in P.items():
        assert len(texts) == 28, (dom, len(texts))
        for i, text in enumerate(texts):
            num = 19 + i                     # 19..46
            cat = f"{dom}_{num:02d}"
            assert cat not in exist_ids
            norm = " ".join(text.lower().split())
            assert norm not in exist_texts, f"dup: {text}"
            assert norm not in protected, f"protected: {text}"
            prompts.append({"category": cat, "prompt": text})
            (train if num <= 36 else val).append(cat)

    assert len(train) == 126 and len(val) == 70
    # within-set text uniqueness
    norms = [" ".join(p["prompt"].lower().split())
             for p in prompts]
    assert len(set(norms)) == 196

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"classification": "STAGE1_SCALE_CORPUS",
         "purpose": "SCALE210 train extension + SCALE-DEV70",
         "splits": {"train": train, "val": val},
         "prompts": prompts}, indent=1))
    import hashlib
    print(f"wrote {OUT}: {len(train)} train + {len(val)} val")
    print("sha256:", hashlib.sha256(
        OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
