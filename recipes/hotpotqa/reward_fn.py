import re
import string
from collections import Counter
from typing import Any

_LOCAL_EM_DATA_SOURCES = {
    "hotpotqa_distractor",
    "2wikimultihopqa",
    "musique",
    "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa",
    "searchR1_musique",
}


def _normalize_answer(s: str) -> str:
    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _extract_answer_from_solution(solution_str: str) -> str:
    """
    Prefer content inside <answer>...</answer>. If not present, fall back to full string.
    """
    pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(pattern, solution_str, flags=re.DOTALL | re.IGNORECASE))
    if not matches:
        return solution_str.strip()
    return matches[-1].group(1).strip()


def _iter_ground_truths(ground_truth: Any) -> list[str]:
    """Convert a ground-truth payload into answer strings.

    Args:
        ground_truth: A string answer, a list/tuple of aliases, or another
            scalar value.

    Returns:
        A list of non-empty answer strings.
    """
    if ground_truth is None:
        return []
    if isinstance(ground_truth, str):
        gt_str = ground_truth.strip()
        return [gt_str] if gt_str else []
    if isinstance(ground_truth, (list, tuple, set)):
        return [str(item).strip() for item in ground_truth if str(item).strip()]
    gt_str = str(ground_truth).strip()
    return [gt_str] if gt_str else []


def _candidate_ground_truths(ground_truth: Any, extra_info: dict | None) -> list[str]:
    """Collect primary and alias answers for EM scoring.

    Args:
        ground_truth: Primary answer from ``reward_model.ground_truth``.
        extra_info: Optional row metadata that may contain ``answers`` aliases.

    Returns:
        Deduplicated answer strings in scoring order.
    """
    candidates = _iter_ground_truths(ground_truth)
    if isinstance(extra_info, dict):
        candidates.extend(_iter_ground_truths(extra_info.get("answers")))

    seen: set[str] = set()
    deduped: list[str] = []
    for answer in candidates:
        if answer in seen:
            continue
        seen.add(answer)
        deduped.append(answer)
    return deduped


def _default_compute_score(*args, **kwargs):
    from verl.utils.reward_score import default_compute_score

    return default_compute_score(*args, **kwargs)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """
    Custom reward function for HotpotQA.

    - If data_source is a HotpotQA-style QA source: use simple exact match
      (EM) between predicted answer and one or more ground-truth answers.
    - Otherwise, fall back to verl's default_compute_score.
    """
    if data_source not in _LOCAL_EM_DATA_SOURCES:
        # Delegate to built-in reward logic for other datasets if any.
        return _default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

    ground_truths = _candidate_ground_truths(ground_truth, extra_info)
    if not ground_truths:
        return 0.0

    pred = _extract_answer_from_solution(solution_str or "")
    norm_pred = _normalize_answer(pred)
    norm_gts = {_normalize_answer(answer) for answer in ground_truths}

    return 1.0 if norm_pred in norm_gts else 0.0


def _answer_f1(pred: str, ground_truths: list[str]) -> float:
    """Return the best SQuAD-style token F1 between a prediction and the ground-truth answers."""
    pred_tokens = _normalize_answer(pred).split()
    best = 0.0
    for answer in ground_truths:
        gold_tokens = _normalize_answer(answer).split()
        common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if common:
            precision = common / len(pred_tokens)
            recall = common / len(gold_tokens)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _answer_in_passages(passages: list[str], ground_truths: list[str]) -> bool:
    """Check whether retrieved text contains an answer. Yes/no never counts: those words appear in any passage."""
    answers = {_normalize_answer(answer) for answer in ground_truths} - {"", "yes", "no"}
    text = f" {_normalize_answer(' '.join(passages))} "
    return any(f" {answer} " in text for answer in answers)


def trajectory_stats(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None,
    passages: list[str],
    queries: list[str],
) -> dict[str, float]:
    """Measure one finished trajectory for reward shaping and behaviour metrics.

    Args:
        data_source: Dataset name of the row.
        solution_str: Decoded response of the trajectory's last step; empty if it ended on a tool call.
        ground_truth: Primary answer from ``reward_model.ground_truth``.
        extra_info: Row metadata that may contain ``answers`` aliases.
        passages: Retrieved passage texts accumulated over the trajectory.
        queries: Search queries issued over the trajectory.

    Returns:
        ``acc`` (exact match, the evaluation metric), ``answered`` (an <answer> tag was given), ``f1`` (0 unless
        answered, so an untagged long response cannot collect token overlap), ``evidence_hit`` and ``repeat_queries``.
    """
    ground_truths = _candidate_ground_truths(ground_truth, extra_info)
    answered = bool(re.search(r"<answer>.*?</answer>", solution_str or "", flags=re.DOTALL | re.IGNORECASE))
    pred = _extract_answer_from_solution(solution_str or "")
    return {
        "acc": float(compute_score(data_source, solution_str, ground_truth, extra_info)),
        "answered": float(answered),
        "f1": _answer_f1(pred, ground_truths) if answered else 0.0,
        "evidence_hit": float(_answer_in_passages(passages, ground_truths)),
        "repeat_queries": float(len(queries) - len({_normalize_answer(query) for query in queries})),
    }
