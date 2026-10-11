from __future__ import annotations

from Agent.data_pipeline.extractors import extract_boxed

from .models import AgentOutput, Prediction


class BoxedLatexOutputAdapter:
    def adapt(self, output: AgentOutput) -> Prediction:
        metadata = dict(output.metadata)
        if output.status != "success":
            return Prediction(output.id, output.status, None, metadata, output.error)
        answer = extract_boxed(output.final_answer or "")
        if answer is None:
            return Prediction(
                output.id,
                "invalid_prediction",
                None,
                metadata,
                {"type": "boxed_answer_missing", "message": "final response has no non-empty boxed answer"},
            )
        return Prediction(output.id, "success", answer.strip(), metadata)
