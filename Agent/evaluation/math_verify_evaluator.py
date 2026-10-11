from __future__ import annotations

from typing import Any, Callable

from .models import Prediction, Reference, SampleResult


class MathVerifyEvaluator:
    def __init__(self, parse_fn: Callable[..., Any] | None = None, verify_fn: Callable[..., bool] | None = None) -> None:
        if parse_fn is None or verify_fn is None:
            try:
                from math_verify import LatexExtractionConfig, parse, verify
            except ImportError as exc:
                raise RuntimeError("Math evaluator requires math-verify[antlr4_13_2]") from exc
            extraction_config = [LatexExtractionConfig()]
            self._parse = lambda value: parse(value, extraction_config=extraction_config)
            self._verify = verify
        else:
            self._parse = parse_fn
            self._verify = verify_fn

    def evaluate(self, prediction: Prediction, reference: Reference) -> SampleResult:
        base = dict(
            id=prediction.id,
            prediction=prediction.answer,
            reference=reference.reference,
            evaluation_scope="development",
            split="validation",
            metadata=dict(reference.metadata),
        )
        if prediction.status != "success" or prediction.answer is None:
            return SampleResult(status=prediction.status, score=0.0, correct=False, error=prediction.error, **base)
        try:
            gold = self._parse(_boxed(reference.reference))
        except Exception as exc:
            return SampleResult(status="evaluator_error", score=0.0, correct=False, error=_error(exc, "gold_parse_error"), **base)
        if not gold:
            return SampleResult(status="evaluator_error", score=0.0, correct=False, error={"type": "gold_parse_error", "message": "empty parsed gold"}, **base)
        try:
            candidate = self._parse(_boxed(prediction.answer))
        except Exception as exc:
            return SampleResult(status="invalid_prediction", score=0.0, correct=False, error=_error(exc, "prediction_parse_error"), **base)
        if not candidate:
            return SampleResult(status="invalid_prediction", score=0.0, correct=False, error={"type": "prediction_parse_error", "message": "empty parsed prediction"}, **base)
        try:
            correct = bool(self._verify(gold, candidate))
        except Exception as exc:
            return SampleResult(status="evaluator_error", score=0.0, correct=False, error=_error(exc, "verification_error"), **base)
        return SampleResult(status="correct" if correct else "incorrect", score=float(correct), correct=correct, error=None, **base)


def _boxed(value: str) -> str:
    return rf"\boxed{{{value.strip()}}}"


def _error(exc: Exception, error_type: str) -> dict[str, str]:
    return {"type": error_type, "message": str(exc)}
