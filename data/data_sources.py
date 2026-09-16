# Data source plan for the 110M pretraining mix (see pretraining-data-spec.md).
# All repos/prefixes/keys verified against the live HF trees on 2026-09-15.
#
# Targets are token targets from the spec; we fetch POOL x the target, measured
# in characters at CHARS_PER_TOKEN (SmolLM2 measured chars/token).

CHARS_PER_TOKEN = 3.93
POOL = 2.5
MIN_TEXT_CHARS = 20
SHARD_UNCOMPRESSED_BYTES = 250_000_000
MIN_FREE_DISK = 5_000_000_000  # stop everything below this


def E(id, repo, prefixes, keys, domain, phases, max_files=None, fmt="auto",
      msgs=None, exclude=None):
    """prefixes: str or list of path prefixes. fmt: auto|parquet|gz|zst|jsonl|jsonarray.
    msgs: (role_key, value_key) when keys[0] is a list-of-messages column.
    exclude: substrings; matching paths are skipped."""
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return dict(id=id, repo=repo, prefixes=prefixes, keys=keys, domain=domain,
                phases=phases, max_files=max_files, fmt=fmt, msgs=msgs,
                exclude=exclude)


ENTRIES = [
    # ---------------- phase 1 ----------------
    # Web, 670M
    E("web-fineweb-edu-10bt", "HuggingFaceFW/fineweb-edu", "sample/10BT/",
      ["text"], "web", [("p1", 670)]),

    # Wikipedia, 130M + phase-2 wiki refresh 40M (same pool, sequential split)
    E("wiki-finewiki-en", "HuggingFaceFW/finewiki", "data/enwiki/",
      ["text"], "wikipedia", [("p1", 130), ("p2", 40)]),

    # Code, 120M. python-edu dropped: its parquet has only blob metadata, no
    # content column. Python code is covered by codeparrot-clean.
    E("code-codeparrot-clean", "codeparrot/codeparrot-clean", "",
      ["content"], "code", [("p1", 50)], max_files=60),

    # Math, 110M
    E("math-finemath-3plus", "HuggingFaceTB/finemath", "finemath-3plus/",
      ["text"], "math", [("p1", 70)]),
    E("math-open-web-math", "open-web-math/open-web-math", "data/",
      ["text"], "math", [("p1", 20)], max_files=15),
    E("math-automathtext-arxiv", "math-ai/AutoMathText",
      ["data/arxiv/0.8", "data/arxiv/0.9"], ["text"], "math", [("p1", 20)],
      max_files=12),

    # Synthetic prose, 80M (+ the 30M phase-2 share of openstax/stanford/khan)
    E("synth-cosmopedia-web-v2", "HuggingFaceTB/cosmopedia", "data/web_samples_v2/",
      ["text"], "synthetic", [("p1", 25)]),
    E("synth-cosmopedia-stories", "HuggingFaceTB/cosmopedia", "data/stories/",
      ["text"], "synthetic", [("p1", 20)]),
    E("synth-cosmopedia-openstax", "HuggingFaceTB/cosmopedia", "data/openstax/",
      ["text"], "synthetic", [("p1", 10), ("p2", 10)]),
    E("synth-cosmopedia-stanford", "HuggingFaceTB/cosmopedia", "data/stanford/",
      ["text"], "synthetic", [("p1", 10), ("p2", 10)]),
    E("synth-cosmopedia-wikihow", "HuggingFaceTB/cosmopedia", "data/wikihow/",
      ["text"], "synthetic", [("p1", 10)]),
    E("synth-cosmopedia-khanacademy", "HuggingFaceTB/cosmopedia", "data/khanacademy/",
      ["text"], "synthetic", [("p1", 5), ("p2", 10)]),

    # Books, 150M
    E("books-project-gutenberg", "common-pile/project_gutenberg", "v0/",
      ["text"], "books", [("p1", 55)]),
    E("books-pre-1929", "common-pile/pre_1929_books", "data/",
      ["text"], "books", [("p1", 40)]),
    E("books-doab", "common-pile/doab", "",
      ["text"], "books", [("p1", 30)]),
    E("books-library-of-congress", "common-pile/library_of_congress", "data/",
      ["text"], "books", [("p1", 25)], max_files=10),

    # Papers, 80M
    E("papers-pes2o", "common-pile/peS2o", "v0/",
      ["text"], "papers", [("p1", 40)]),
    E("papers-arxiv-abstracts", "common-pile/arxiv_abstracts", "",
      ["text"], "papers", [("p1", 20)]),
    E("papers-proofpile2-arxiv", "EleutherAI/proof-pile-2", "arxiv/train/",
      ["text"], "papers", [("p1", 20)], max_files=4),

    # QA & forums, 80M
    E("qa-stackexchange", "common-pile/stackexchange", "",
      ["text"], "qa", [("p1", 50)], max_files=150, exclude=[".meta."]),
    E("qa-dolmino-stackexchange", "allenai/dolmino-mix-1124", "data/stackexchange/",
      ["text"], "qa", [("p1", 15)]),
    E("qa-ubuntu-irc", "common-pile/ubuntu_irc", "v0/",
      ["text"], "qa", [("p1", 15)]),

    # Micro-domains, 80M
    E("micro-uspto", "common-pile/uspto", "",
      ["text"], "micro", [("p1", 15)], max_files=2),
    E("micro-caselaw", "common-pile/caselaw_access_project", "",
      ["text"], "micro", [("p1", 15)], max_files=12),
    E("micro-regulations", "common-pile/regulations", "v0/",
      ["text"], "micro", [("p1", 10)]),
    E("micro-usgpo", "common-pile/usgpo", "data/",
      ["text"], "micro", [("p1", 10)], max_files=6),
    E("micro-youtube", "common-pile/youtube", "",
      ["text"], "micro", [("p1", 10)], max_files=6),
    E("micro-foodista", "common-pile/foodista", "v0/documents/",
      ["text"], "micro", [("p1", 10)]),
    E("micro-pep", "common-pile/python_enhancement_proposals", "v0/documents/",
      ["text"], "micro", [("p1", 5)]),
    E("micro-bhl", "common-pile/biodiversity_heritage_library", "v0/",
      ["text"], "micro", [("p1", 5)], max_files=3),

    # ---------------- phase 2 ----------------
    # Instructional / high-quality prose, 100M (cosmopedia share lives in the
    # synth entries above via multi-phase pools)
    E("anneal-dolmino-flan", "allenai/dolmino-mix-1124", "data/flan/",
      ["text"], "anneal_instructional", [("p2", 40)], max_files=12),
    E("anneal-webinstructsub", "TIGER-Lab/WebInstructSub", "data/",
      ["question", "answer"], "anneal_instructional", [("p2", 30)], max_files=6),

    # Math quality, 90M
    E("anneal-finemath-4plus", "HuggingFaceTB/finemath", "finemath-4plus/",
      ["text"], "anneal_math_quality", [("p2", 50)]),
    E("anneal-automathtext-web", "math-ai/AutoMathText",
      ["data/web/0.8", "data/web/0.9"], ["text"], "anneal_math_quality",
      [("p2", 25)], max_files=6),
    E("anneal-infiwebmath-4plus", "HuggingFaceTB/finemath", "infiwebmath-4plus/",
      ["text"], "anneal_math_quality", [("p2", 15)], max_files=2),

    # Code quality, 80M
    E("anneal-opc-algorithmic", "OpenCoder-LLM/opc-annealing-corpus",
      "algorithmic_corpus/", ["text"], "anneal_code_quality", [("p2", 20)], max_files=4),
    E("anneal-opc-snippets", "OpenCoder-LLM/opc-annealing-corpus",
      "synthetic_code_snippet/", ["text"], "anneal_code_quality", [("p2", 20)], max_files=4),
    E("anneal-opc-synthetic-qa", "OpenCoder-LLM/opc-annealing-corpus",
      "synthetic_qa/", ["text"], "anneal_code_quality", [("p2", 10)], max_files=2),
    E("anneal-opencodereasoning", "nvidia/OpenCodeReasoning", "",
      ["input", "output"], "anneal_code_quality", [("p2", 30)], max_files=8),

    # Math solutions, 70M
    E("anneal-openmathinstruct-2", "nvidia/OpenMathInstruct-2", "data/train-",
      ["problem", "generated_solution"], "anneal_math_solutions", [("p2", 40)], max_files=4),
    E("anneal-openr1-220k", "open-r1/OpenR1-Math-220k", "all/train-",
      ["problem", "solution"], "anneal_math_solutions", [("p2", 30)], max_files=20),

    # Wiki + QA refresh, 70M (wiki 40M comes from the wiki-finewiki-en entry)
    E("anneal-stackexchange-filtered", "common-pile/stackexchange_filtered", "",
      ["text"], "anneal_wiki_qa", [("p2", 30)], max_files=3),

    # OER / OCR books, 40M
    E("anneal-libretexts", "common-pile/libretexts_filtered", "",
      ["text"], "anneal_oer_ocr", [("p2", 10)]),
    E("anneal-oercommons", "common-pile/oercommons_filtered", "",
      ["text"], "anneal_oer_ocr", [("p2", 10)]),
    E("anneal-olmocr-s2pdf", "allenai/olmOCR-mix-0225", "train-s2pdf",
      ["response"], "anneal_oer_ocr", [("p2", 10)], max_files=1),
    E("anneal-olmocr-iabooks", "allenai/olmOCR-mix-0225", "train-iabooks",
      ["response"], "anneal_oer_ocr", [("p2", 5)], max_files=1),
    E("anneal-pressbooks", "common-pile/pressbooks_filtered", "",
      ["text"], "anneal_oer_ocr", [("p2", 5)]),

    # Reasoning traces, 25M
    E("anneal-openthoughts", "open-thoughts/OpenThoughts-114k", "data/",
      ["conversations"], "anneal_reasoning", [("p2", 25)], max_files=3,
      msgs=("from", "value")),

    # Agentic, 25M
    E("agentic-swe-agent", "nebius/SWE-agent-trajectories", "data/",
      ["trajectory"], "anneal_agentic", [("p2", 8)], max_files=4,
      msgs=("role", "text")),
    E("agentic-commitpackft", "bigcode/commitpackft", "data/",
      ["message", "new_contents"], "anneal_agentic", [("p2", 8)], max_files=150),
    E("agentic-glaive", "glaiveai/glaive-function-calling-v2", "",
      ["chat"], "anneal_agentic", [("p2", 5)], max_files=1, fmt="jsonarray"),
    E("agentic-openhands", "SWE-Gym/OpenHands-Sampled-Trajectories", "data/train.raw-",
      ["messages"], "anneal_agentic", [("p2", 4)], max_files=3,
      msgs=("role", "content")),
]

# Noted exclusions for the manifest (sources we know we cannot fetch as specced)
MISSING_NOTES = {
    "HuggingFaceTB/smollm-corpus#python-edu":
        "parquet carries only blob metadata (blob_id/path/score), no content "
        "column; python code budget shifted to codeparrot-clean",
}
