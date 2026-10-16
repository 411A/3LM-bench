#!/usr/bin/env python3
"""
Second half of the task build: MC/knowledge, Persian, networking, DevOps.

Design decisions traceable to docs/METHODOLOGY.md:
  * MC items are answered with ONE forward pass via logprobs (zero generated
    tokens) - at 3-8 tok/s this is the difference between 3 min and 3 hours.
  * Persian scoring needs normalisation: Arabic->Persian letters, digit
    scripts, ZWNJ, NFKC. Naive exact-match destroys the data.
  * Networking + DevOps follow FaulT-Bench's majority-counterexample design:
    40% of items are FALSE-PREMISE (nothing is actually broken). Without
    those you measure compliance, not diagnosis.
  * DevOps uses FDE-Bench's gated 0/0.25/0.5/0.75/1.0 lattice with no LLM judge.
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import debias_mc  # noqa: E402

OUT = os.path.join(HERE, "tasks")
rnd = random.Random(20261012)

# Every multiple-choice file goes through the position de-bias on the way out, so
# the de-biased file IS the build output. De-biasing used to be a separate manual
# step that no builder called, so `python3 build_tasks2.py` regenerated the
# 69%-on-B suite - the exact defect it was written to remove.
MC_FILES = {"science", "persian", "english", "electronics"}


def dump(name, rows):
    if name in MC_FILES:
        debias_mc.rebalance(rows)
        hist, best = debias_mc.report(rows)
        print(f"       (gold positions {hist}, best-constant baseline {best:.1%})")
    p = os.path.join(OUT, name + ".json")
    json.dump(rows, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"  {name:<26} {len(rows):>4} tasks -> {p}")


# ------------------------------------------------------------------ science
# GPQA deliberately EXCLUDED: 198 items where PhD experts score 65%; a 4B model
# is indistinguishable from the 25% random baseline. ARC-C + MMLU STEM subsets
# still discriminate at 2-4B.
SCIENCE = [
    # ARC-Challenge style
    # "immediately" discriminated nothing: Transduction and Transformation are both
    # bacterial DNA transfer, so both were defensible. Now names the vector.
    ("arc-1", "Which process transfers bacterial DNA from one bacterium to another via a bacteriophage?",
     ["Transduction", "Translation", "Transcription", "Transformation"], 0, "biology"),
    ("arc-2", "A gas is compressed isothermally to half its volume. What happens to its pressure?",
     ["Doubles", "Halves", "Stays the same", "Quadruples"], 0, "physics"),
    ("arc-3", "Which particle carries the electromagnetic force?",
     ["Gluon", "Photon", "W boson", "Graviton"], 1, "physics"),
    ("arc-4", "In a controlled experiment, the independent variable is the factor that:",
     ["Is measured as the outcome", "Is deliberately manipulated", "Is kept constant", "Is inferred statistically"], 1, "biology"),
    ("arc-5", "An element in group 17 of the periodic table is most likely to:",
     ["Gain electrons and form -1 ions", "Lose electrons and form +1 ions", "Share three electrons", "Form metallic bonds"], 0, "chemistry"),
    ("arc-6", "Which of these is a pure substance?",
     ["Seawater", "Air", "Distilled water", "Steel"], 2, "chemistry"),
    ("arc-7", "A wave with a frequency of 50 Hz has a period of:",
     ["0.02 s", "2 s", "50 s", "0.5 s"], 0, "physics"),
    ("arc-8", "In DNA, adenine pairs with:",
     ["Guanine", "Uracil", "Cytosine", "Thymine"], 3, "biology"),
    ("arc-9", "Which best explains why metals conduct electricity?",
     ["Free electrons in the lattice", "High melting point", "Ionic bonds", "Neutron scattering"], 0, "chemistry"),
    ("arc-10", "The speed of light in vacuum is approximately:",
     ["3 x 10^8 m/s", "3 x 10^6 m/s", "3 x 10^10 m/s", "3 x 10^4 m/s"], 0, "physics"),
    ("arc-11", "Natural selection acts most directly on:",
     ["Genotypes", "Phenotypes", "Individual alleles only", "Populations of species only"], 1, "biology"),
    ("arc-12", "Which change is chemical rather than physical?",
     ["Ice melting", "Iron rusting", "Sugar dissolving", "Water boiling"], 1, "chemistry"),
    ("arc-13", "In an ecosystem, organisms that make their own food are called:",
     ["Consumers", "Producers", "Decomposers", "Predators"], 1, "biology"),
    ("arc-14", "The SI unit of electrical resistance is the:",
     ["Volt", "Ampere", "Ohm", "Watt"], 2, "physics"),
    ("arc-15", "A solution with pH 3 is:",
     ["Basic", "Acidic", "Neutral", "Saline"], 1, "chemistry"),
    # MMLU computer science
    # "average ... in the worst case" mixes two different complexities and is
    # self-contradictory. One case only.
    ("cs-1", "What is the time complexity of quicksort in the worst case?",
     ["O(n)", "O(log n)", "O(n log n)", "O(n^2)"], 3, "computer_science"),
    ("cs-2", "Which data structure gives O(1) average lookup by key?",
     ["Linked list", "Binary tree", "Hash table", "Sorted array"], 2, "computer_science"),
    ("cs-3", "In a binary search tree, an in-order traversal yields keys in:",
     ["Random order", "Descending order", "Ascending order", "Level order"], 2, "computer_science"),
    ("cs-4", "What does the 'S' in ACID stand for in databases?",
     ["Speed", "Scalability", "Atomicity", "Synchronisation"], 2, "computer_science"),
    # "in-place" was dropped: no listed algorithm is stable AND in-place AND
    # O(n log n). Merge sort is stable and O(n log n) but needs O(n) auxiliary
    # space, so the item rewarded the "merge sort is the stable one" heuristic
    # and punished the model that knows merge sort is not in-place.
    ("cs-5", "Which sorting algorithm is stable and runs in O(n log n) average time, "
     "at the cost of O(n) auxiliary space?",
     ["Quicksort", "Merge sort", "Heapsort", "Selection sort"], 1, "computer_science"),
    ("cs-6", "A deadlock requires which conditions?",
     ["Mutual exclusion only", "Mutual exclusion, hold and wait, no preemption, circular wait",
      "Preemption and circular wait", "Single-threaded execution"], 1, "computer_science"),
    ("cs-7", "What does a compiler do during semantic analysis?",
     ["Strip whitespace", "Check type and name semantics", "Link symbols", "Optimise machine code"], 1, "computer_science"),
    # Segregated fit ALSO suffers external fragmentation (it only mitigates it), so
    # it was a second defensible answer. Replaced with a scheme that genuinely
    # does not: buddy allocation splits and coalesces fixed-size blocks, so a
    # free block can always be merged back into the parent.
    ("cs-8", "Which memory allocation strategy suffers from external fragmentation?",
     ["First fit", "Buddy system", "Slab allocation", "Fixed-partition paging"], 0, "computer_science"),
    ("cs-9", "In Big-O terms, binary search on a sorted array of n items is:",
     ["O(1)", "O(log n)", "O(n)", "O(n log n)"], 1, "computer_science"),
    ("cs-10", "A hash table with chaining and a good hash has expected lookup:",
     ["O(1)", "O(log n)", "O(n)", "O(n^2)"], 0, "computer_science"),
    # MMLU physics
    ("ph-1", "A block slides down a frictionless incline. Its acceleration depends on:",
     ["The incline angle", "The block's mass", "The block's colour", "The incline length only"], 0, "physics"),
    ("ph-2", "Newton's second law states that force equals:",
     ["mass x acceleration", "mass / acceleration", "acceleration / mass", "mass x velocity"], 0, "physics"),
    ("ph-3", "Which quantity is conserved in an elastic collision but not in an inelastic one?",
     ["Momentum", "Kinetic energy", "Mass", "Charge"], 1, "physics"),
    ("ph-4", "The escape velocity from a planet depends on:",
     ["Only the planet's mass", "The planet's mass and radius", "The mass of the object", "The object's colour"], 1, "physics"),
    ("ph-5", "Ohm's law relates voltage to:",
     ["current and resistance", "charge and time", "mass and acceleration", "energy and power"], 0, "physics"),
    # MMLU chemistry
    ("ch-1", "Avogadro's number is approximately:",
     ["6.02 x 10^23", "3.14 x 10^8", "1.60 x 10^-19", "9.11 x 10^-31"], 0, "chemistry"),
    ("ch-2", "An exothermic reaction:",
     ["absorbs heat", "releases heat", "is always reversible", "requires a catalyst"], 1, "chemistry"),
    ("ch-3", "The number of moles in 22.4 L of gas at STP is approximately:",
     ["0.5", "1.0", "2.0", "22.4"], 1, "chemistry"),
    ("ch-4", "A catalyst affects a reaction by:",
     ["Shifting equilibrium", "Lowering activation energy", "Increasing enthalpy", "Consuming reactants"], 1, "chemistry"),
    ("ch-5", "Which bond type involves electron sharing?",
     ["Ionic", "Covalent", "Metallic", "Coordinate only"], 1, "chemistry"),
]


def science():
    return [{"id": f"sci/{i}", "source": "ARC-C/MMLU-style hand-authored",
             "category": "science_mc", "lang": "en",
             "question": q, "choices": c, "answer": a, "subject": s}
            for i, q, c, a, s in SCIENCE]


# ------------------------------------------------------------------ Persian
# Normalisation is mandatory: Arabic yeh/kaf, Persian vs Arabic-Indic digits,
# ZWNJ, NFKC. See normalize_fa.py in scorers.
PERSIAN = [
    ("fa-mc-1", "کدام یک از گزینه‌های زیر یک عدد اول است؟",
     ["۴", "۶", "۷", "۹"], 2),
    ("fa-mc-2", "پایتخت ایران کدام شهر است؟",
     ["شیراز", "تهران", "اصفهان", "تبریز"], 1),
    ("fa-mc-3", "بزرگ‌ترین سیاره منظومه شمسی کدام است؟",
     ["زهره", "مریخ", "مشتری", "زمین"], 2),
    ("fa-mc-4", "گاز اصلی جو زمین چیست؟",
     ["اکسیژن", "نیتروژن", "کربن دی اکسید", "هیدروژن"], 1),
    ("fa-mc-5", "اگر ۲۵ درصد عددی برابر ۸۰ باشد، آن عدد چقدر است؟",
     ["۲۰۰", "۳۲۰", "۴۰۰", "۲۵۰"], 1),
    ("fa-mc-6", "کدام یک یک واحد اندازه‌گیری مقاومت الکتریکی است؟",
     ["ولت", "آمپر", "اهم", "وات"], 2),
    # This item used to ask which digit SCRIPT Persian writes 0-9 in, with the gold
    # "۰ تا ۹" appearing verbatim in the question, and three of the four choices
    # (Persian, Arabic-Indic, ASCII) collapsing to one string under the suite's
    # own normalize_fa. That is unrepairable in place: normalize_fa treats
    # Arabic-Indic and Persian digits as equivalent because they are semantically
    # equivalent in Persian text, so a question whose whole point is that
    # distinction contradicts the scorer.
    # First replacement quoted the gold term («پایگاه داده») in the stem, which
    # is the same leak. The stem now gives the ENGLISH gloss and never quotes a
    # choice, so the item cannot be answered by copying.
    ("fa-mc-7", "کدام گزینه معادل انگلیسی «database» است؟",
     ["پایگاه داده", "پایگاه اتصال", "درگاه عبور", "شبکهٔ عبور"], 0),
    ("fa-mc-8", "کدام زبان یک زمان راست‌به‌چپ است؟",
     ["انگلیسی", "فارسی", "فرانسه", "آلمانی"], 1),

    # ---------------------------------------------------------------------
    # The suite was 14 items. At n=14 a model must score 57% before the data
    # can rule out guessing at all (stats.py), so the suite was close to
    # incapable of measuring anything - a model at 43% was indistinguishable
    # from a coin flip, and 2 items is the entire margin between those states.
    # Expanded toward the size of the science suite. Every arithmetic gold here
    # is re-derived independently in verify_golds.py.
    # ---------------------------------------------------------------------
    ("fa-mc-9", "سردترین سیارهٔ منظومهٔ شمسی کدام است؟",
     ["مریخ", "زهرا", "نپتون", "جوبیتر"], 3),
    ("fa-mc-10", "شیراز در کدام استان قرار دارد؟",
     ["اصفهان", "تهران", "فارس", "خراسان"], 2),
    ("fa-mc-11", "واحد اندازه‌گیری نیرو کدام است؟",
     ["ژول", "نیوتن", "وات", "پاسکال"], 1),
    ("fa-mc-12", "شیمی‌دان ایرانی که کتاب «کشف‌الاسرار» را نوشت، که بود؟",
     ["ابوریحان", "دانشمند نکته‌ای", "محمد بن ابراهیم", "بیرون"], 0),
    ("fa-mc-13", "در گیاهان، رنگ سبز از کدام رنگدانه است؟",
     ["کاروتنوئید", "کلروفیل", "آنتوسیانین", "بیلی‌روبین"], 1),
    ("fa-mc-14", "کدام عدد اول است؟",
     ["۲۱", "۲۷", "۲۹", "۳۳"], 2),
    ("fa-mc-15", "در سیستم متریک، هر کیلوگرم برابر چند گرم است؟",
     ["۱۰", "۱۰۰", "۱۰۰۰", "۱۰۰۰۰"], 2),
    ("fa-mc-16", "«می‌روم» از نظر دستوری چه نوعی است؟",
     ["اسم", "فعل", "صفت", "حرف اضافه"], 1),
    ("fa-mc-17", "قارهٔ کوچک‌تر کدام است؟",
     ["اروپا", "آفریقا", "استرالیا", "آمریکای جنوبی"], 2),
    ("fa-mc-18", "مولکول آب از چند اتم ساخته شده است؟",
     ["۲", "۳", "۴", "۵"], 1),
    ("fa-mc-19", "کدام زبان برنامه‌نویسی هم به‌عنوان نام یک جنس پوستهٔ سخت‌تنگه نیز به کار می‌رود؟",
     ["پایتون", "جاوا", "پی اچ پی", "گو"], 0),
    ("fa-mc-20", "در بدن انسان، کدام اندام وظیفهٔ پمپاژ خون را دارد؟",
     ["ریه", "کبد", "قلب", "کلیه"], 2),
    ("fa-mc-21", "عدد ۹ در فارسی چگونه نوشته می‌شود؟",
     ["شش", "هشت", "نه", "ده"], 2),
    ("fa-mc-22", "کدام یک از این‌ها یک گاز نجیب است؟",
     ["اکسیژن", "آرگون", "نیتروژن", "کربن‌دی‌اکسید"], 1),
    ("fa-mc-23", "در تبدیل واحد دما، ۱۰۰ درجهٔ سانتی‌گراد برابر چند درجهٔ فارنهایت است؟",
     ["۲۱۲", "۱۰۰", "۳۲", "۴۰۰"], 0),
    ("fa-mc-24", "انسان معمولاً چند دندان دائمی دارد؟",
     ["۱۶", "۲۰", "۳۲", "۶۴"], 2),
    ("fa-math-4", "حاصل ۱۲ ضرب در ۵ چقدر است؟",
     ["۵۰", "۶۰", "۷۰", "۸۰"], 1),
    ("fa-math-5", "حاصل ۲۴۰ تقسیم بر ۸ چقدر است؟",
     ["۲۰", "۲۵", "۳۰", "۴۰"], 2),
    ("fa-math-6", "اگر قیمت کالایی ۵۰ هزار تومان باشد و ۲۰ درصد تخفیف بخورد، قیمت نهایی چقدر است؟",
     ["۳۰ هزار", "۴۰ هزار", "۴۵ هزار", "۶۰ هزار"], 1),
    # "how much did it fall" was ambiguous without the final price given - the
    # answer depends on an unstated starting point.
    ("fa-math-7", "قیمت کالایی از ۱۰۰ هزار تومان به ۷۵ هزار تومان رسید؛ "
     "این کاهش چند درصد است؟",
     ["۱۵ درصد", "۲۰ درصد", "۲۵ درصد", "۴۰ درصد"], 2),
    ("fa-math-8", "حاصل ۱۰۰ منهای ۳۷ چقدر است؟",
     ["۵۳", "۶۳", "۶۷", "۷۳"], 1),
    ("fa-math-9", "اگر ۴ کتاب ۶۰ هزار تومان باشد، یک کتاب چند تومان است؟",
     ["۱۰ هزار", "۱۲ هزار", "۱۵ هزار", "۲۰ هزار"], 2),
    ("fa-math-10", "میانگین اعداد ۲، ۴، ۶ و ۸ چقدر است؟",
     ["۴", "۵", "۶", "۷"], 1),
    ("fa-mc-25", "کدام یک از این‌ها یک عدد زوج است؟",
     ["۷", "۹", "۱۲", "۱۵"], 2),
    ("fa-mc-26", "در سیستم‌های دیجیتال، هر بیت چند مقدار می‌تواند داشته باشد؟",
     ["یک", "دو", "هشت", "شانزده"], 1),
    ("fa-mc-27", "کدام سیاره بیشترین تعداد قمر دارد؟",
     ["زمین", "مریخ", "زحل", "عطارد"], 2),
    # Labels are now DEFINED, and deliberately NOT named in the question. The first
    # attempt defined them using the choice strings themselves ("«بی‌ربط» یعنی
    # ..."), which put the gold verbatim into every stem - the same leak the
    # control suite exists to catch, introduced by the fix for the leak. The
    # options are therefore numeric and the stem describes the three cases
    # without quoting any option.
    #
    # Convention, now stated explicitly:
    #   1 - the hypothesis necessarily follows from the premise (entailment)
    #   2 - the hypothesis contradicts the premise
    #   3 - the premise neither entails nor contradicts it (neutral)
    #
    # fa-nli-1 was golded 2: "همه دانشجویان حاضرند" from "بعضی دانشجویان حاضرند".
    # Under the convention above that is 3, not a contradiction - "some" is
    # compatible with "all". Corrected, which also makes all three consistent.
    ("fa-nli-1",
     "برای هر جمله یکی از سه گزینه را انتخاب کنید:\n"
     "۱ = نتیجه حتماً از مقدم به دست می‌آید.\n"
     "۲ = نتیجه با مقدم تناقض دارد.\n"
     "۳ = مقدم نه نتیجه را تأیید می‌کند و نه رد می‌کند.\n\n"
     "مقدم: «بعضی دانشجویان در کلاس حضور داشتند.»\n"
     "نتیجه: «همه دانشجویان در کلاس حضور داشتند.»",
     ["۱", "۲", "۳"], 2),
    ("fa-nli-2",
     "برای هر جمله یکی از سه گزینه را انتخاب کنید:\n"
     "۱ = نتیجه حتماً از مقدم به دست می‌آید.\n"
     "۲ = نتیجه با مقدم تناقض دارد.\n"
     "۳ = مقدم نه نتیجه را تأیید می‌کند و نه رد می‌کند.\n\n"
     "مقدم: «علی در خانه بود.»\n"
     "نتیجه: «علی کتاب خواند.»",
     ["۱", "۲", "۳"], 2),
    ("fa-nli-3",
     "برای هر جمله یکی از سه گزینه را انتخاب کنید:\n"
     "۱ = نتیجه حتماً از مقدم به دست می‌آید.\n"
     "۲ = نتیجه با مقدم تناقض دارد.\n"
     "۳ = مقدم نه نتیجه را تأیید می‌کند و نه رد می‌کند.\n\n"
     "مقدم: «آسمان آبی است.»\n"
     "نتیجه: «هوا آبی است.»",
     ["۱", "۲", "۳"], 2),
    # 80000 / 5 = 16000 per book; 16000 * 3 = 48000 = "۴۸ هزار".
    # The gold was ۲۴۰ هزار = 80000*3, i.e. the "forgot to divide by 5" error, and
    # the correct 48,000 was not among the choices at all. A live 4B model picked
    # the wrong gold at 0.997 confidence, so the item rewarded the exact mistake
    # small models make and penalised the correct reasoning.
    # The distractors are now the characteristic errors instead: x3 without
    # dividing, x2, and divisor-as-multiplier. The arithmetic is written out here
    # on purpose - verify_golds.py re-derives it, so a future edit cannot silently
    # reintroduce this class of defect.
    ("fa-math-1", "اگر پنج کتاب ۸۰ هزار تومان باشد، سه کتاب چند تومان است؟",
     ["۴۸ هزار", "۲۴۰ هزار", "۱۶۰ هزار", "۴۰۰ هزار"], 0),
    ("fa-math-2", "حاصل ۱۷ ضرب در ۳ چقدر است؟",
     ["۴۰", "۴۵", "۵۰", "۵۱"], 3),
    ("fa-math-3", "اگر قیمت کالایی ۲۰۰ هزار تومان باشد و ۱۰ درصد تخفیف بخورد، قیمت نهایی چقدر است؟",
     ["۱۶۰ هزار", "۱۷۰ هزار", "۱۸۰ هزار", "۲۰۰ هزار"], 2),
]


def persian():
    rows = []
    for i, q, ch, a in PERSIAN:
        cat = "fa_nli" if "-nli-" in i else ("fa_math" if "-math-" in i else "fa_mc")
        rows.append({"id": f"fa/{i}", "source": "hand-authored/Persian",
                     "category": cat, "lang": "fa",
                     "question": q, "choices": ch, "answer": a, "subject": cat})
    return rows


# ------------------------------------------------------------------ English MC
ENGLISH = [
    ("en-mc-1", "Which sentence is grammatically correct?",
     ["He don't like it.", "She doesn't likes it.", "She doesn't like it.", "They doesn't like it."], 2),
    ("en-mc-2", "'Fewer' is correctly used with:",
     ["water", "items", "advice", "information"], 1),
    ("en-mc-3", "The past participle of 'write' is:",
     ["wrote", "written", "writing", "writed"], 1),
    ("en-mc-4", "Which word is a synonym for 'enormous'?",
     ["tiny", "colossal", "average", "fragile"], 1),
    ("en-mc-5", "Identify the logical fallacy: 'Everyone I know is voting for him, so he must be right.'",
     ["Straw man", "Bandwagon", "False dilemma", "Circular reasoning"], 1),
    ("en-mc-6", "What is the main idea of the sentence: 'Despite the delay, the team completed the migration on time.'",
     ["The migration was cancelled", "The team finished on schedule despite problems",
      "The team was late", "The migration took longer than expected"], 1),
    # Malformed analogy: "A compiler is to software as a:" is ungrammatical and
    # missing its second term, and none of the options is a valid counterpart to
    # "software" - a compiler is itself software. Replaced with a well-formed
    # item with a single defensible answer.
    ("en-mc-7", "A compiler translates which kind of text into machine code?",
     ["source code", "hardware", "operating system", "a browser engine"], 0),
    ("en-mc-8", "Which is the correct order of magnitude for a typical laptop CPU clock speed?",
     ["100 kHz", "1 MHz", "1 GHz", "1 THz"], 2),
    ("en-mc-9", "In an English sentence, the subject is the:",
     ["verb", "noun or noun phrase doing the action", "adverb", "conjunction"], 1),
    ("en-mc-10", "'Despite' expresses:",
     ["causation", "concession", "sequence", "condition"], 1),
    ("en-mc-11", "TCP guarantees which property?",
     ["Low latency", "Reliable ordered delivery", "Broadcast to all hosts", "Fixed bandwidth"], 1),
    ("en-mc-12", "Which HTTP status code means 'Not Found'?",
     ["200", "301", "404", "500"], 2),
    ("en-mc-13", "A hash table's worst-case lookup is:",
     ["O(1)", "O(log n)", "O(n)", "O(n log n)"], 2),
    ("en-mc-14", "What does DNS translate?",
     ["IP to MAC", "Domain names to IP addresses", "Ports to services", "Certificates to keys"], 1),
    ("en-mc-15", "In a git workflow, which command records staged changes?",
     ["git add", "git commit", "git push", "git pull"], 1),

    # ---------------------------------------------------------------------
    # Expanded from 15 items for the same reason as the Persian suite: at n=15 a
    # model must score 53% before guessing can be excluded (stats.py), so the
    # suite could not separate a 2B model from a 4B model. Targets ~40.
    # ---------------------------------------------------------------------
    ("en-mc-16", "Which of these is a symmetric-key cipher?",
     ["RSA", "AES", "ECC", "Diffie-Hellman"], 1),
    ("en-mc-17", "What is the time complexity of binary search on a sorted array of n items?",
     ["O(1)", "O(log n)", "O(n)", "O(n log n)"], 1),
    ("en-mc-18", "Which HTTP status code means 'Unauthorized'?",
     ["400", "401", "403", "404"], 1),
    ("en-mc-19", "In Big-O terms, what is the average lookup cost of a hash table with a good hash and chaining?",
     ["O(1)", "O(log n)", "O(n)", "O(n log n)"], 0),
    ("en-mc-20", "Which of these is a strongly-typed language?",
     ["JavaScript", "Python", "TypeScript", "Ruby"], 2),
    ("en-mc-21", "In a relational database, which clause filters grouped results?",
     ["WHERE", "HAVING", "ORDER BY", "JOIN"], 1),
    ("en-mc-22", "What does DNS primarily do?",
     ["Assign IP addresses", "Map names to IP addresses",
      "Encrypt web traffic", "Route packets"], 1),
    ("en-mc-23", "Which of these best describes a race condition?",
     ["Two threads interleaving so the result depends on timing",
      "A process waiting for input",
      "A deadlock between two threads",
      "A buffer that is too small"], 0),
    ("en-mc-24", "What is the result of 7 // 2 in Python?",
     ["3.5", "3", "4", "An error"], 1),
    ("en-mc-25", "Which principle states that a network service should never see a partial transaction?",
     ["Atomicity", "Redundancy", "Sharding", "Memoization"], 0),
    ("en-mc-26", "In machine learning, overfitting means the model",
     ["performs well on training data and poorly on unseen data",
      "cannot converge during training",
      "has too many features to load into memory",
      "produces probabilities that are too low"], 0),
    ("en-mc-27", "Which of the following is NOT a valid Python identifier?",
     ["_private", "class_name", "2fast", "myVar"], 2),
    ("en-mc-28", "What does the 'S' in HTTPS stand for?",
     ["Secure", "Standard", "Simple", "Session"], 0),
    ("en-mc-29", "In Big-O notation, which grows fastest?",
     ["O(n)", "O(n log n)", "O(n^2)", "O(n^2 log n)"], 3),
    ("en-mc-30", "Which data structure gives O(1) insert and delete at the front?",
     ["A singly linked list", "A deque", "A binary heap", "A binary search tree"], 1),
    ("en-mc-31", "In Git, what does a merge conflict indicate?",
     ["The local branch is ahead of the remote",
      "Overlapping changes could not be reconciled automatically",
      "The working tree has uncommitted files",
      "The repository has no commits yet"], 1),
    ("en-mc-32", "Which of these is a side effect of calling a function?",
     ["Returning a value", "Modifying a global list in place",
      "Computing a local variable", "Evaluating an argument"], 1),
    ("en-mc-33", "What is the default port for HTTPS?",
     ["21", "22", "80", "443"], 3),
    ("en-mc-34", "Which sorting algorithm has a guaranteed O(n log n) worst case?",
     ["Quicksort", "Heapsort", "Bubble sort", "Insertion sort"], 1),
    ("en-mc-35", "In a hash table, what is a collision?",
     ["Two keys mapping to the same slot",
      "A key being too long",
      "A hash function being too slow",
      "A table being too small for the key type"], 0),
    ("en-mc-36", "Which HTTP method is intended to replace a resource?",
     ["GET", "POST", "PUT", "PATCH"], 2),
    ("en-mc-37", "What does 'tail call optimisation' avoid?",
     ["Recursive calls growing the stack",
      "Using global variables",
      "Calling functions with many arguments",
      "Allocating heap memory"], 0),
    ("en-mc-38", "Which of these best describes a race in distributed systems?",
     ["Two nodes disagree about which commit won",
      "A node taking too long to respond",
      "A network partition",
      "A disk filling up"], 0),
    ("en-mc-39", "In unit testing, a 'mock' is used to",
     ["replace a dependency with a controlled stand-in",
      "measure elapsed time",
      "generate random test data",
      "profile memory usage"], 0),
    ("en-mc-40", "What is the purpose of an index in a relational database?",
     ["To speed up row lookup at the cost of extra storage and write time",
      "To enforce referential integrity",
      "To compress the table",
      "To normalise the schema"], 0),
    ("en-math-1", "What is 12 x 5?", ["50", "60", "70", "80"], 1),
    ("en-math-2", "What is 240 / 8?", ["20", "25", "30", "40"], 2),
    ("en-math-3", "A price falls from $100 to $75. What percentage decrease is that?",
     ["15%", "20%", "25%", "40%"], 2),
    ("en-math-4", "What is the average of 2, 4, 6 and 8?", ["4", "5", "6", "7"], 1),
    ("en-math-5", "If 4 books cost $60, one book costs how much?",
     ["$10", "$12", "$15", "$20"], 2),
]


def english():
    return [{"id": f"en/{i}", "source": "hand-authored/English", "category": "english_mc",
             "lang": "en", "question": q, "choices": c, "answer": a, "subject": "english"}
            for i, q, c, a in ENGLISH]


# ------------------------------------------------------------------ networking
# 40% false-premise, per FaulT-Bench: the network is HEALTHY but the ticket
# claims a fault. Correct answer is is_anomaly=false. Without these you
# measure compliance rather than diagnosis.
NETDIAG = [
    # (topology, config, symptom, observed, is_anomaly, device, root_cause, fix)
    # net/0: the gold cause was "dns" with no DNS evidence anywhere in the config
    # or the observation - "router pingable, internet unreachable" equally
    # supports NAT or a default-route fault. The observation now carries the DNS
    # evidence that the gold claims.
    ("2 hosts 1 switch",
     "host-a: ip 10.0.0.10/24 gw 10.0.0.1\nhost-b: ip 10.0.0.20/24 gw 10.0.0.1\nrouter: 10.0.0.1/24",
     "Users cannot reach the internet",
     "ping 10.0.0.1 from host-a: replies; ping 8.8.8.8: replies; dig example.com: no response from any resolver", True,
     "router", "dns", "configure DNS on the router"),
    # net/1: the faulty device was golded as r2, but the bad prefix is r1's - a
    # /30 on 10.0.1.5 covers 10.0.1.4-7 and does NOT contain 10.0.1.1, which is
    # the address r1's own default route points at. A model that correctly blamed
    # r1 lost the device third.
    ("3-router OSPF",
     "r1: interface eth1 ip address 10.0.1.5/30\nr1: ip route 0.0.0.0/0 via 10.0.1.1\nr2: interface eth1 ip address 10.0.1.6/24",
     "Hosts in 10.0.0.0/24 intermittently lose the default gateway",
     "ping 10.0.1.1 from r1: Destination Host Unreachable; ping 10.0.1.1 from r2: replies", True,
     "r1", "subnet_mask", "widen the prefix on r1 eth1 so 10.0.1.1 is on-link"),
    ("VLAN 200 + trunk",
     "switch1: vlan 200\nswitch1: interface gi0/1 switchport trunk allowed vlan 10,20\nswitch2: interface gi0/1 switchport access vlan 200",
     "VLAN 200 hosts cannot reach the gateway", "show interface trunk: vlan 200 not allowed", True,
     "switch1", "vlan", "add vlan 200 to the trunk allowed list"),
    ("DHCP scope",
     "server: scope 10.0.5.0 255.255.255.0\nserver: option 3 10.0.5.1\nclients: 10.0.5.50 gw 10.0.5.1 dns 10.0.5.1",
     "Clients get an IP but cannot resolve names", "nslookup google.com: server 10.0.5.1 timeout", True,
     "client", "dns", "point clients at a working resolver"),
    ("BGP session",
     "r1: router bgp 65001\nr1: neighbor 10.0.2.2 remote-as 65002\nr2: router bgp 65002\nr2: neighbor 10.0.2.1 remote-as 65001",
     "BGP routes vanished after a reboot", "show ip bgp summary: neighbor 10.0.2.2 Idle", True,
     "r1", "bgp_session", "re-establish the BGP session"),
    ("healthy - false premise",
     "r1: interface eth0 ip address 10.0.0.1/24\nr1: ip route 0.0.0.0/0 via 192.168.1.1\nclients: dhcp from 10.0.0.1",
     "IT ticket: 'users report total network outage since 09:00'",
     "ping 10.0.0.1 ok, ping 8.8.8.8 ok, DNS resolves", False,
     None, "none", "no action - network is healthy, gather more evidence"),
    ("healthy - false premise 2",
     "host: ip 192.168.1.50/24 gw 192.168.1.1\nhost: dns 192.168.1.1\nrouter: default route present",
     "IT ticket: 'DNS is broken on the subnet'",
     "dig example.com returns a valid A record", False,
     None, "none", "no action - DNS resolution verified working"),
    ("healthy - false premise 3",
     "switch: vlan 10, vlan 20 configured\ntrunk allows vlan 10,20,30\nhost in vlan 20 with correct gw",
     "IT ticket: 'switch is dropping all VLAN traffic'",
     "show interfaces: all up, no errors; hosts ping gateway", False,
     None, "none", "no action - counters show healthy traffic"),
    ("healthy - false premise 4",
     "r1: interface eth0 up\nr1: interface eth1 down (no cable)\nusers on eth0 have connectivity",
     "IT ticket: 'r1 interface eth1 is down, please fix'",
     "show ip int brief: eth1 administratively down, interface eth0 up", False,
     None, "none", "eth1 is intentionally unused; eth0 path is healthy"),
    # net/9: the config said r2's address was "10.0.0.5/24 (duplicate of r1)"
    # while the ARP table implies a duplicate of r1's 10.0.0.1. 10.0.0.5 !=
    # 10.0.0.1, so the config did not show the conflict the item is about.
    ("wrong device",
     "r1 healthy: 10.0.0.1/24\nr2: interface eth0 ip address 10.0.0.1/24 (duplicate of r1)",
     "Ticket says 'r1 has a bad subnet mask'",
     "arp -a shows 10.0.0.1 MAC belongs to r2", True,
     "r2", "address_conflict", "remove the duplicate address on r2"),
    # net/10: the symptom said "cannot reach 8.8.8.8" while the observation showed
    # 8.8.8.8 replying - self-contradictory. The gold is a DNS cause, so the
    # symptom now names resolution.
    ("wrong cause",
     "host: ip 10.0.0.10/24 gw 10.0.0.1\nhost: /etc/resolv.conf nameserver 9.9.9.9\nrouter: NAT enabled",
     "Hosts cannot resolve domain names",
     "ping 8.8.8.8 from host: replies; dig example.com: SERVFAIL", True,
     "host", "dns", "fix the nameserver configuration"),
]


def netdiag():
    return [{"id": f"net/{i}", "source": "hand-authored/FaulT-Bench-lineage",
             "category": "net_diag", "lang": "en",
             "topology": t, "config": c, "symptom": s, "observed": o,
             "is_anomaly": a, "faulty_device": d, "root_cause": r, "fix": f}
            for i, (t, c, s, o, a, d, r, f) in enumerate(NETDIAG)]


# ------------------------------------------------------------------ DevOps
# FDE-Bench's gated lattice {0,.25,.5,.75,1} and no-LLM-judge rule, with the
# execution layers replaced by static semantic checks.
DEVOPS = [
    ("dockerfile-fastapi", "Write a Dockerfile that runs a FastAPI app on port 8000, as a non-root user, with a healthcheck. Output only Dockerfile content.",
     {"must_contain": ["FROM", "EXPOSE 8000", "USER", "HEALTHCHECK"],
      "must_not_contain": ["USER root", "privileged"], "lang": "dockerfile"}),
    ("dockerfile-multistage", "Write a multi-stage Dockerfile that builds a Go binary and copies it into a minimal final image. Output only Dockerfile content.",
     {"must_contain": ["FROM", "AS", "COPY"], "must_not_contain": [], "lang": "dockerfile"}),
    ("compose-nginx-postgres", "Write a docker-compose file for an nginx service and a postgres 15 service with a persistent volume. Output only YAML.",
     {"must_contain": ["services:", "nginx", "postgres", "volumes:"],
      "must_not_contain": ["privileged: true"], "lang": "compose"}),
    ("compose-healthcheck", "Write a docker-compose with an app service that has a depends_on a db service which defines a healthcheck. Output only YAML.",
     {"must_contain": ["services:", "depends_on", "healthcheck"],
      "must_not_contain": [], "lang": "compose"}),
    ("k8s-deployment-service", "Write a Kubernetes Deployment and Service exposing a web app on port 80 with a readiness probe and non-root security context. Output only YAML.",
     {"must_contain": ["kind: Deployment", "kind: Service", "readinessProbe", "runAsNonRoot"],
      "must_not_contain": ["image: nginx:latest"], "lang": "k8s"}),
    ("k8s-labels-match", "Write a Kubernetes Deployment where selector.matchLabels exactly equals template.metadata.labels. Output only YAML.",
     {"must_contain": ["kind: Deployment", "matchLabels", "labels"], "must_not_contain": [], "lang": "k8s"}),
    ("cicd-github-actions", "Write a GitHub Actions workflow that runs tests on push to main using a pinned action version. Output only YAML.",
     {"must_contain": ["on:", "jobs:", "runs-on:", "uses:"],
      "must_not_contain": ["@master", "@main", "write-all"], "lang": "cicd"}),
    ("cicd-secrets", "Write a GitHub Actions workflow that reads a token from secrets and does NOT print it. Output only YAML.",
     {"must_contain": ["secrets.", "if:", "run:"], "must_not_contain": ["echo ${{ secrets"], "lang": "cicd"}),
    ("systemd-unit", "Write a systemd service unit for a Python app that restarts on failure. Output only the unit file.",
     {"must_contain": ["[Service]", "ExecStart", "Restart="], "must_not_contain": [], "lang": "systemd"}),
    ("nginx-vhost", "Write an nginx server block that proxies /api to a backend on port 8000 and serves static files. Output only the config.",
     {"must_contain": ["server {", "location /api", "proxy_pass", "listen"],
      "must_not_contain": [], "lang": "nginx"}),
]


def devops():
    return [{"id": f"devops/{i}", "source": "hand-authored/FDE-Bench-lineage",
             "category": "devops", "lang": i2, "prompt": p, "checks": ch}
            for i, (i2, p, ch) in enumerate(DEVOPS)]


if __name__ == "__main__":
    print("building task set 2...")
    dump("science", science())
    dump("persian", persian())
    dump("english", english())
    dump("netdiag", netdiag())
    dump("devops", devops())
    print("done")
