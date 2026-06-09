from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.vllm.constant import TEMPLATE_MAP
from tests.model_serving.model_runtime.vllm.probes.utils import VLLM_LIVENESS_PROBE, VLLM_READINESS_PROBE
from utilities.constants import Containers, RuntimeTemplates
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
) -> Generator[ServingRuntime, Any, Any]:
    """vLLM ServingRuntime with readiness and liveness httpGet probes on kserve-container."""
    accelerator_type = supported_accelerator_type.lower()
    template_name = TEMPLATE_MAP.get(accelerator_type, RuntimeTemplates.VLLM_CUDA)
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param["deployment_type"],
        runtime_image=vllm_runtime_image,
        support_tgis_open_ai_endpoints=True,
        containers={
            Containers.KSERVE_CONTAINER_NAME: {
                "readinessProbe": VLLM_READINESS_PROBE,
                "livenessProbe": VLLM_LIVENESS_PROBE,
            }
        },
    ) as model_runtime:
        yield model_runtime
