import os
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from pytest import FixtureRequest
from syrupy.extensions.json import JSONSnapshotExtension

from tests.model_serving.model_runtime.vllm.modelcar.constant import MODELCAR_REGISTRIES
from tests.model_serving.model_runtime.vllm.modelcar.utils import (
    build_registry_pull_secret_string_data,
    get_registry_key_for_host,
    normalize_registry_pull_auth,
    registry_host_from_oci_uri,
    validate_registry_pull_auth,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    modelcar_group = parser.getgroup(name="Modelcar")
    modelcar_group.addoption(
        "--redhat-registry-pull-secret",
        default=os.environ.get("REDHAT_REGISTRY_PULL_SECRET"),
        help="Base64 auth or JSON creds with 'auth' for registry.redhat.io (required for modelcar tests)",
    )
    modelcar_group.addoption(
        "--quay-oci-pull-secret",
        default=os.environ.get("QUAY_OCI_PULL_SECRET"),
        help="Base64 auth or JSON creds with 'auth' for quay.io (optional)",
    )
    modelcar_group.addoption(
        "--stage-registry-pull-secret",
        default=os.environ.get("STAGE_REGISTRY_PULL_SECRET"),
        help="Base64 auth or JSON creds with 'auth' for registry.stage.redhat.io (optional)",
    )


def _resolve_registry_pull_auth(raw_value: str | None, expected_host: str) -> str | None:
    if not raw_value:
        return None
    auth = normalize_registry_pull_auth(raw_value=raw_value, expected_host=expected_host)
    validate_registry_pull_auth(auth=auth)
    return auth


@pytest.fixture(scope="session")
def redhat_registry_pull_auth(pytestconfig: pytest.Config) -> str:
    """Return base64 auth for registry.redhat.io (mandatory for modelcar tests)."""
    raw_value = pytestconfig.getoption(name="--redhat-registry-pull-secret")
    if not raw_value:
        raise ValueError(
            "Red Hat registry pull secret is not set. "
            "Either pass --redhat-registry-pull-secret or set REDHAT_REGISTRY_PULL_SECRET"
        )
    auth = _resolve_registry_pull_auth(
        raw_value=raw_value,
        expected_host=MODELCAR_REGISTRIES["redhat"]["host"],
    )
    assert auth is not None
    return auth


@pytest.fixture(scope="session")
def quay_oci_pull_auth(pytestconfig: pytest.Config) -> str | None:
    """Return base64 auth for quay.io when provided."""
    return _resolve_registry_pull_auth(
        raw_value=pytestconfig.getoption("--quay-oci-pull-secret"),
        expected_host=MODELCAR_REGISTRIES["quay"]["host"],
    )


@pytest.fixture(scope="session")
def stage_registry_pull_auth(pytestconfig: pytest.Config) -> str | None:
    """Return base64 auth for registry.stage.redhat.io when provided."""
    return _resolve_registry_pull_auth(
        raw_value=pytestconfig.getoption("--stage-registry-pull-secret"),
        expected_host=MODELCAR_REGISTRIES["stage"]["host"],
    )


@pytest.fixture(scope="class")
def modelcar_registry_pull_secrets(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    redhat_registry_pull_auth: str,
    quay_oci_pull_auth: str | None,
    stage_registry_pull_auth: str | None,
) -> Generator[list[Secret], Any, Any]:
    """Create one pull secret per configured registry; redhat is always created."""
    storage_uri = request.param.get("model_car_image_uri", "")
    if storage_uri:
        image_host = registry_host_from_oci_uri(storage_uri=storage_uri)
        registry_key = get_registry_key_for_host(host=image_host)
        if registry_key is None:
            raise ValueError(f"Unsupported OCI registry host: {image_host}")

        auth_by_key: dict[str, str | None] = {
            "redhat": redhat_registry_pull_auth,
            "quay": quay_oci_pull_auth,
            "stage": stage_registry_pull_auth,
        }
        required_auth = auth_by_key.get(registry_key)
        if required_auth is None:
            env_var = MODELCAR_REGISTRIES[registry_key]["env_var"]
            raise ValueError(
                f"Model image uses {image_host} but pull secret is not set. "
                f"Set {env_var} or pass the matching CLI option."
            )

    auth_entries: list[tuple[str, str]] = [("redhat", redhat_registry_pull_auth)]
    if quay_oci_pull_auth:
        auth_entries.append(("quay", quay_oci_pull_auth))
    if stage_registry_pull_auth:
        auth_entries.append(("stage", stage_registry_pull_auth))

    secrets: list[Secret] = []
    with ExitStack() as stack:
        for registry_key, auth in auth_entries:
            registry = MODELCAR_REGISTRIES[registry_key]
            secret = stack.enter_context(
                cm=Secret(
                    client=admin_client,
                    name=registry["secret_name"],
                    namespace=model_namespace.name,
                    string_data=build_registry_pull_secret_string_data(host=registry["host"], auth=auth),
                    type="kubernetes.io/dockerconfigjson",
                    wait_for_resource=True,
                )
            )
            secrets.append(secret)
        yield secrets


@pytest.fixture
def response_snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)
