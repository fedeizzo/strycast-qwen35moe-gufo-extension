# strycast-qwen35moe-gufo-extension

A static [Gufo](https://github.com/gufo-org/gufo) extension that serves
**Qwen3.6-35B-A3B** and **Ornith 1.5 122B** (GGUF architecture `qwen35moe`) on
AMD Strix Halo (gfx1151), with MTP and DFlash2 speculative decoding. It is the
35B runner from [Strycast](https://github.com/slimsami/strycast) packaged as an
external extension, built against Gufo `main` with no change to Gufo itself.

**Status: early.** It builds against Gufo `main` `6cc8a42c` and gives the same
outputs as Strycast. It isn't released or versioned yet, and it is not checked
against later Gufo commits.

## Build

Clone Gufo `main` and point its build at this directory:

```sh
cmake --preset gpu-test -DGUFO_EXTENSION_DIRS=/absolute/path/to/strycast-qwen35moe-gufo-extension
cmake --build --preset gpu-test --target gufo --parallel
build/gpu-test/gufo diagnose --section extensions --json
```

`diagnose` should list the extension `strycast_qwen35moe` with provider
`qwen35moe` available. It builds beside Gufo's own extensions
(`-DGUFO_EXTENSIONS=*` works) and only claims `gguf:qwen35moe`, so Gufo's
`qwen27-q4` and `flash-next-q4` providers are unaffected. Linux only, ROCm build
only: the providers need HIP.

## Serve

```sh
gufo serve --sessions 4 llm --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --context 262144 \
  --speculative mtp --mtp-model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --draft-tokens 2
```

For DFlash2, use `--speculative dflash2 --dflash-model
Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf --draft-tokens 7 --draft-policy adaptive`.
The supported GGUFs, memory use and benchmarks are in Strycast's
[35B guide](https://github.com/slimsami/strycast/tree/main/docs/models/qwen3.6-35b-a3b)
and [Ornith guide](https://github.com/slimsami/strycast/tree/main/docs/models/ornith-1.5-122b).

## Checked so far

On Strix Halo (ROCm 10.1), against Strycast `0fd9bf59` built with the same
toolchain, greedy outputs are byte-identical:

- 35B Q8 and Q4_K_M, plain, DFlash2 and MTP, at 0, 8K and 32K context;
- 1-4 concurrent users with DFlash2 and MTP;
- Ornith 1.5 122B Q4_K_M, plain and MTP.

Speed is within noise everywhere. With repetitive prompts at 4 users, both
builds are bimodal (for example 128 or 144 tok/s with MTP), depending on whether
one request starts decoding before the others.

## How it is made

Strycast stays the source for now. `tools/generate.sh` rebuilds this tree from a
Strycast checkout:

1. It applies `tools/upstream-drift.patch`, Gufo's Qwen changes after
   Strycast's base commit, to a temporary Strycast worktree.
2. `make_spike.py` copies the Qwen family code (`q35/family`), the Qwen model
   and runner (`q35/model`) and the Flash-Next kernels the 35B uses
   (`q35/fnext`). It rewrites private includes to Gufo's public `<gufo/ext/...>`
   headers using `ext-public-map.txt`.
3. `rename_ns.py` moves the copy from `gufo::` to `strycast::`, and the C-linkage
   MMQ symbols to a `q35_` prefix. The copy is linked into the same binary as
   Gufo's own Qwen libraries, so it can't share their names.
4. The result is committed as "generated", then `tools/ports/*.patch` are
   applied on top. These patches adapt Strycast code to the public API:

| Patch | Why |
| --- | --- |
| 0001 | Gufo's `ModelConfig` has no routed-expert fields; the extension derives its own. |
| 0002 | No hook for the MTP prime chunk size; it uses the default 1,024. |
| 0003 | No hook for the DFlash2 context skip or the prefill-step checkpoint grid; both are dropped. |
| 0004 | No per-step draft cap in the verifier; the cap for concurrent sessions on MoE targets moves into the draft backends, with identical proposals. |
| 0005 | Build fixes against Gufo `main` (moved headers, a private test helper). |

```sh
tools/generate.sh /path/to/strycast /tmp/out 0fd9bf59
```

The tree in this repository is exactly that output.

## License

MIT, see `LICENSE`. Third-party notices are in `NOTICE` and `licenses/`.
