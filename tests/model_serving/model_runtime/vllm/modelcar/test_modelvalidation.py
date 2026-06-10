from typing import Any

import pytest
import structlog
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.utils import (
    validate_raw_openai_inference_request,
)
from tests.model_serving.model_runtime.vllm.modelcar.constant import COMPLETION_QUERY

LOGGER = structlog.get_logger(name=__name__)


pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type")


@pytest.mark.model_validation
class TestVLLMModelCarRaw:
    def test_oci_model_car_raw_openai_inference(
        self,
        vllm_model_car_inference_service: InferenceService,
        response_snapshot: Any,
        deployment_config: dict[str, Any],
    ) -> None:
        """Validate OpenAI inference over the external route for a vLLM model served from an OCI modelcar image."""
        LOGGER.info("Sending inference request to vLLM model served from OCI image via external route.")
        validate_raw_openai_inference_request(
            isvc=vllm_model_car_inference_service,
            model_name=vllm_model_car_inference_service.instance.metadata.name,
            response_snapshot=response_snapshot,
            completion_query=COMPLETION_QUERY,
            model_output_type=deployment_config.get("model_output_type"),
        )
