import pytest

from tests.model_serving.model_server.utils import (
    verify_inference_response,
)
from utilities.constants import ModelFormat, ModelStoragePath, Protocols
from utilities.inference_utils import Inference
from utilities.manifests.caikit_tgis import CAIKIT_TGIS_INFERENCE_CONFIG


@pytest.mark.parametrize(
    "unprivileged_model_namespace, unprivileged_s3_caikit_serverless_inference_service",
    [
        pytest.param(
            {"name": "non-admin-serverless"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL_CAIKIT},
        )
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.serverless
class TestServerlessUnprivilegedUser:
    @pytest.mark.polarion("ODS-2552")
    def test_non_admin_deploy_serverless_and_query_model(self, unprivileged_s3_caikit_serverless_inference_service):
        """Verify non admin can deploy a model and query using REST"""
        verify_inference_response(
            inference_service=unprivileged_s3_caikit_serverless_inference_service,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, unprivileged_s3_caikit_raw_inference_service",
    [
        pytest.param(
            {"name": "non-admin-raw"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL_HF},
        )
    ],
    indirect=True,
)
@pytest.mark.sanity
@pytest.mark.rawdeployment
class TestRawUnprivilegedUser:
    @pytest.mark.polarion("ODS-2611")
    def test_non_admin_deploy_raw_and_query_model(
        self,
        unprivileged_s3_caikit_raw_inference_service,
    ):
        """Verify non admin can deploy a Raw model and query using REST"""
        verify_inference_response(
            inference_service=unprivileged_s3_caikit_raw_inference_service,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )
