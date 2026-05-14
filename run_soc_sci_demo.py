import argparse
import sys
import time

from ollama_http import DEFAULT_BASE_URL, OllamaError, chat, pick_gemma_model
from soc_sci_tasks import TASKS, get_task_keys


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--task", default="all", choices=["all", *get_task_keys()])
    p.add_argument("--base-url", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    return p.parse_args()


def _print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def main():
    args = _parse_args()
    base_url = args.base_url or DEFAULT_BASE_URL
    try:
        model = pick_gemma_model(preferred=args.model, base_url=base_url)
    except OllamaError as e:
        print(str(e), file=sys.stderr)
        return 2

    selected = []
    if args.task == "all":
        selected = [TASKS[k] for k in get_task_keys()]
    else:
        selected = [TASKS[args.task]]

    print(f"model={model}")
    for idx, spec in enumerate(selected, start=1):
        _print_section(f"任务 {idx}: {spec.title}  ({spec.key})")
        print("【System Prompt】")
        print(spec.system_prompt)
        print("\n【User Input】")
        print(spec.user_input)
        print("-" * 80)
        print("【Model Output】\n")
        start_time = time.time()
        max_tokens = args.max_tokens if args.max_tokens is not None else (spec.max_tokens or 900)
        out = chat(
            model=model,
            system=spec.system_prompt,
            user=spec.user_input,
            temperature=spec.temperature,
            top_p=spec.top_p,
            max_tokens=max_tokens,
            base_url=base_url,
        )
        if not out:
            retry_tokens = max(max_tokens * 4, 2000)
            out = chat(
                model=model,
                system=spec.system_prompt + "\n只输出最终答案，不要输出思考过程。",
                user=spec.user_input,
                temperature=spec.temperature,
                top_p=spec.top_p,
                max_tokens=retry_tokens,
                base_url=base_url,
            )
            if not out:
                print(f"（空输出：可尝试增大 --max-tokens，例如 {retry_tokens} 或更高）")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(out)
        print("\n" + "-" * 80)
        print(f"【运行时长】: {elapsed_time:.2f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

