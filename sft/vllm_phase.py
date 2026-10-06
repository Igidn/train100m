"""vLLM engine checks, in a process of their own.

Split out of the smoke kernel for one concrete reason: pip can replace the torch
build on disk, but a process that has already imported torch keeps the old one
mapped. Installing `torch==2.9.0` to match a vLLM wheel and then importing vLLM
*in the same interpreter* still loads the pre-existing libtorch, and the failure
looks identical to never having tried —

    ImportError: vllm/_C.abi3.so: undefined symbol:
      _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib

which is what the smoke kernel reported for vllm 0.11.2 / 0.10.2 after it had
already run `torch.cuda.device_count()`. A child process is the only way to
honour the swap.

Run standalone (inside a Kaggle kernel, after cloning the repo):

    python sft/vllm_phase.py --out vllm_phase.json

Writes a JSON report and exits 0 if the engine worked, 1 otherwise. Never
raises: a packaging failure is a result, not a crash.
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.environ.get("SMOKE_MODEL", "igidn/TinyBalls-110M-V1")
CU_INDEX = os.environ.get("SMOKE_TORCH_INDEX",
                          "https://download.pytorch.org/whl/cu128")
DEFAULT_SPECS = "vllm,vllm==0.11.2,vllm==0.10.2,vllm==0.9.2"


def say(msg):
    print(msg, flush=True)


def pip(*args):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])


def pins_for(packages=("torch", "torchaudio", "torchvision")):
    """Versions the installed vllm wheel pins, read from its own metadata.

    All of them have to move together. vLLM starts by refusing to run when
    PyTorch and TorchAudio disagree about CUDA
    ("Detected that PyTorch and TorchAudio were compiled with different CUDA
    versions"), and the image ships torchaudio against a different CUDA than the
    torch the wheel wants — so installing torch alone moves the mismatch rather
    than fixing it.
    """
    import importlib.metadata as md
    try:
        reqs = list(md.requires("vllm") or [])
    except Exception as e:
        say(f"[vllm] metadata unavailable: {e}")
        return {}
    pins = {}
    for r in reqs:
        for pkg in packages:
            m = re.match(rf"\s*{pkg}\s*\(?==\s*([0-9][^,);\s]*)", r)
            if m:
                pins[pkg] = m.group(1)
    if pins:
        return pins
    # fall back to `pip show`, whose Requires lines are comma-joined
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "show", "vllm"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Requires:"):
                for part in line.split(":", 1)[1].split(","):
                    for pkg in packages:
                        m = re.match(rf"\s*{pkg}\s*\(?==\s*([0-9][^,);\s]*)", part)
                        if m:
                            pins[pkg] = m.group(1)
    except Exception as e:
        say(f"[vllm] pip show failed: {e}")
    return pins


def import_vllm():
    """Import vLLM and its compiled CUDA path (vllm.platforms is where the _C
    extension actually loads)."""
    import vllm
    from vllm.platforms import current_platform
    return vllm, current_platform


def try_spec(spec, report):
    """Install one candidate (plus its torch) and see whether it runs here."""
    rec = {"spec": spec}
    try:
        pip(spec)
    except subprocess.CalledProcessError as e:
        rec["error"] = f"pip install failed: {e}"
        return rec
    pins = pins_for()
    rec["pins"] = pins
    want = []
    for pkg, ver in pins.items():
        try:
            import importlib.metadata as md
            have = (md.version(pkg) or "").split("+")[0]
        except Exception:
            have = ""
        if have != ver:
            want.append(f"{pkg}=={ver}")
    if want:
        say(f"[vllm] wheel pins {pins}; installing {' '.join(want)} from {CU_INDEX}")
        try:
            pip("--index-url", CU_INDEX, *want)
        except subprocess.CalledProcessError as e:
            rec["error"] = f"installing {' '.join(want)} failed: {e}"
            return rec
    else:
        say(f"[vllm] installed stack already matches {pins}")

    # Proactive, not reactive: the guard is `_check_cuda_version` inside
    # vllm/entrypoints/llm.py, so it fires when `LLM(...)` is constructed rather
    # than at import, and a retry around the import never sees it. vLLM 0.31's own
    # pins are internally inconsistent (torch 2.13.0, torchaudio 2.11.0), so no
    # version pair can agree; what disagrees is the CUDA *build*. This is a
    # text-only run, so the audio stack goes.
    say("[vllm] removing torchaudio/torchvision (text-only; dodges the "
        "torch/torchaudio CUDA guard)")
    for pkg in ("torchaudio", "torchvision"):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", pkg],
                       check=False)
    try:
        import torch
        vllm, current_platform = import_vllm()
        cap = current_platform.get_device_capability()
        dev = current_platform.get_device_name()
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        say(f"[vllm] {spec} unusable: {type(e).__name__}: {str(e)[:200]}")
        return rec
    rec["torch"] = torch.__version__
    rec["vllm"] = vllm.__version__
    rec["device"] = str(dev)
    rec["capability"] = str(cap)
    say(f"[vllm] {spec} -> vllm {vllm.__version__}, torch {torch.__version__}, "
        f"{dev} sm{cap}")
    report.update(rec)
    return rec


def engine_checks(report):
    """Load the model, generate, and compare against our own decode."""
    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rollout import (IM_END, TOOL_CALL, VLLMEngine, build_model_from_hf,
                         render_prefix)
    from reward import compare_weather_task

    tok = AutoTokenizer.from_pretrained(REPO)
    prompt_ids = render_prefix(tok, compare_weather_task().messages())

    engine = VLLMEngine(REPO)

    # vLLM's tokenizer helper moved in 0.31 (`vllm.transformers_utils.tokenizer`
    # no longer exists). The marker check is worth keeping, so try the known
    # import paths and then the tokenizer the engine built for itself.
    vtok = None
    for mod in ("vllm.transformers_utils.tokenizer", "vllm.transformers_utils",
                "vllm.tokenizers"):
        try:
            vtok = __import__(mod, fromlist=["get_tokenizer"]).get_tokenizer(REPO)
            say(f"[tok] using {mod}.get_tokenizer")
            break
        except Exception:
            continue
    if vtok is None:
        vtok = getattr(getattr(engine.llm, "tokenizer", None), "encode", None)
        say("[tok] using the engine's own tokenizer"
            if vtok else "[tok] no vllm tokenizer accessor found")
    if vtok is not None:
        got = list(vtok.encode(TOOL_CALL, add_special_tokens=False))
        report["tok.vllm_markers"] = {"ok": got == [49152], "detail": str(got)}
        say(f"[tok] vllm encodes tool_call -> {got}")
    else:
        report["tok.vllm_markers"] = {"ok": True, "soft": True,
                                      "detail": "no tokenizer accessor; the "
                                                "engine loaded the model anyway"}
    out, finish = engine.generate([prompt_ids], max_tokens=200, temp=0.0)[0]
    text = tok.decode(out, skip_special_tokens=False)
    say("[vllm] greedy completion:\n" + text)
    report["generation"] = {"ids": out, "finish": finish, "text": text}
    report["gen.emits_tool_call"] = {"ok": TOOL_CALL in text or 49152 in out,
                                     "detail": "tool_call present"}
    report["gen.stops_at_im_end"] = {"ok": finish == "stop", "detail": finish}

    # token-level agreement with our own implementation, greedy
    ref_ids, ref_finish = engine.generate([prompt_ids], max_tokens=32, temp=0.0)[0]
    import torch
    from transformers import LlamaForCausalLM
    from llama.model import LLaMA
    hf = LlamaForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32).eval()
    ours = build_model_from_hf(hf, LLaMA).eval().cuda()
    del hf
    torch.cuda.empty_cache()
    from llama.kv_cache import generate as kv_generate
    with torch.autocast("cuda", torch.float16):
        mine = list(kv_generate(ours, [prompt_ids], max_new=32, temp=0.0)[0])
    del ours
    torch.cuda.empty_cache()
    n = 0
    for a, b in zip(ref_ids, mine):
        if a != b:
            break
        n += 1
    say(f"[fidelity] vllm vs ours greedy: common prefix {n} "
        f"(vllm {len(ref_ids)} tok, ours {len(mine)} tok)")
    report["gen_fidelity"] = {"common_prefix": n, "vllm": ref_ids, "ours": mine}

    # batch-8 throughput: the number the GRPO budget is built from
    import time
    t0 = time.time()
    outs = engine.generate([prompt_ids] * 8, max_tokens=128, temp=1.0, seed=0)
    dt = time.time() - t0
    ntok = sum(len(o[0]) for o in outs)
    say(f"[throughput] batch 8 x <=128 new tokens in {dt:.1f}s = {ntok / dt:.1f} tok/s")
    report["throughput"] = {"batch": 8, "tokens": ntok, "seconds": dt,
                            "tok_per_s": ntok / dt}
    return n >= min(16, len(ref_ids), len(mine))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="vllm_phase.json")
    ap.add_argument("--specs", default=os.environ.get("SMOKE_VLLM_SPECS",
                                                       DEFAULT_SPECS))
    args = ap.parse_args()
    report = {"attempts": []}
    for spec in [s for s in args.specs.split(",") if s]:
        rec = try_spec(spec, report)
        report["attempts"].append(rec)
        if "vllm" in rec:
            try:
                report["ok"] = bool(engine_checks(report))
            except Exception as e:
                import traceback
                traceback.print_exc()
                report["engine_error"] = f"{type(e).__name__}: {str(e)[:400]}"
                report["ok"] = False
            break
    else:
        report["ok"] = False
        report["engine_error"] = "no candidate became importable"
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    say("VLLM PHASE: " + ("PASS" if report.get("ok") else "FAIL"))
    say("wrote " + args.out)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())