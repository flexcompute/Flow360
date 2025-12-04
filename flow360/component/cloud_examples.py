"""Cloud examples interface for fetching and copying Flow360 examples"""

from __future__ import annotations

from typing import List

import pydantic as pd_v2

from flow360.cloud.flow360_requests import CopyExampleRequest
from flow360.cloud.responses import (
    CopyExampleResponse,
    ExampleItem,
    ExamplesListResponse,
)
from flow360.cloud.rest_api import RestApi
from flow360.environment import Env
from flow360.exceptions import Flow360WebError
from flow360.log import log

DRIVAER_ID = "prj-7fb80c26-6565-4ea5-97b6-9bf5e87882f2"


def fetch_examples() -> List[ExampleItem]:
    """
    Fetch available examples from the cloud.

    Returns
    -------
    List[ExampleItem]
        List of available example items.
    """
    api = RestApi("public/v2/examples")
    resp = api.get()
    if resp is None:
        return []
    try:
        response_model = ExamplesListResponse(**resp if isinstance(resp, dict) else {"data": resp})
        return response_model.data
    except (pd_v2.ValidationError, TypeError, ValueError) as e:
        log.warning(f"Failed to parse examples response: {e}")
        return []


def show_available_examples() -> None:
    """
    Display available examples in a formatted table.

    Shows a list of pre-executed project examples that can be copied and visited
    on the Flow360 web interface.
    """
    examples = fetch_examples()
    if not examples:
        print("No examples available.")
        return

    examples_url = Env.current.get_web_real_url("examples")
    print(f"These examples are pre-executed projects that can be visited on {examples_url}")
    print()

    title_width = max(len(e.title) for e in examples)
    id_width = max(len(e.id) for e in examples)

    header = f"{'#':>3}  {'Title'.ljust(title_width)}  {'Example ID'.ljust(id_width)}  Tags"
    print(header)
    print("-" * len(header))

    for idx, ex in enumerate(examples):
        title = ex.title
        example_id = ex.id
        tags = ", ".join(ex.tags)
        print(f"{idx+1:>3}  {title.ljust(title_width)}  {example_id.ljust(id_width)}  {tags}")


def copy_example(example_id: str) -> str:
    """
    Copy an example from the cloud and return the created project ID.

    Parameters
    ----------
    example_id : str
        ID of the example to copy.

    Returns
    -------
    str
        Project ID of the newly created project.

    Raises
    ------
    Flow360WebError
        If the example cannot be copied or the response format is unexpected.
    """
    request = CopyExampleRequest(source_example_id=example_id)
    example_api = RestApi("v2/examples")
    resp = example_api.post(request.dict(), method="copy")
    if not isinstance(resp, dict):
        raise Flow360WebError(f"Unexpected response format when copying example {example_id}")
    response_model = CopyExampleResponse(**resp)
    return response_model.id
