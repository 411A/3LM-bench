# Third-party notices

The top-level MIT license applies to original project code, documentation, and
project-authored task content. The following bundled third-party inputs and
assets carry separate terms.

## HumanEval

The source input `data/res/HumanEval.jsonl.gz` and derived task instances in
`tasks/code_humaneval.json` originate from OpenAI's HumanEval dataset. The
upstream project is licensed under the MIT License:

- Source: <https://github.com/openai/human-eval>
- Paper: Chen et al., *Evaluating Large Language Models Trained on Code*,
  <https://arxiv.org/abs/2107.03374>

## MBPP sanitized

The source input `data/res/sanitized-mbpp.json` and derived task instances in
`tasks/code_mbpp.json` originate from the MBPP dataset and sanitized
distribution. They are identified as CC BY 4.0 in `benchmarks/catalog.json`.
Attribution: Google Research / Austin et al., *Program Synthesis with Large
Language Models*, <https://arxiv.org/abs/2108.07732>. License:
<https://creativecommons.org/licenses/by/4.0/>. The generated task file
reformats prompts, adds harness metadata, and contains the 426-item subset
documented in the catalog; it is an adaptation of the upstream material.

## Apache ECharts

`reports/assets/echarts.min.js` bundles Apache ECharts 5.6.1 and retains its
upstream license and bundled component notices. The matching upstream notice
is included at `reports/assets/NOTICE.txt`; a copy of the Apache License 2.0 is
included at `reports/assets/LICENSE-APACHE-2.0.txt`. ECharts is distributed
under that license:

- Project: <https://github.com/apache/echarts>
- License: <https://www.apache.org/licenses/LICENSE-2.0>
- Upstream notice: <https://github.com/apache/echarts/blob/master/NOTICE>

Other benchmark datasets are fetched from their upstream sources and are not
committed as raw downloads. Their licenses, URLs, and protocol notes are listed
in [`benchmarks/catalog.json`](benchmarks/catalog.json). Respect each upstream
license when acquiring or using those datasets.
