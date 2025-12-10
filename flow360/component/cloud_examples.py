"""Cloud examples interface for fetching and copying Flow360 examples"""

from __future__ import annotations

import difflib
import time
from typing import List, Optional, Tuple

import pydantic as pd_v2

from flow360.cloud.flow360_requests import CopyExampleRequest
from flow360.cloud.responses import (
    CopyExampleResponse,
    ExampleItem,
    ExamplesListResponse,
)
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import ProjectInterface
from flow360.environment import Env
from flow360.exceptions import Flow360Error, Flow360ValueError, Flow360WebError
from flow360.log import log


def find_example_by_name(query_name: str, examples: List[ExampleItem]) -> Tuple[ExampleItem, float]:
    """
    Find the best matching example by name using fuzzy string matching.

    Parameters
    ----------
    query_name : str
        The name to search for (case-insensitive, handles typos).
    examples : List[ExampleItem]
        List of available examples to search through.

    Returns
    -------
    Tuple[ExampleItem, float]
        The best matching example and its similarity score (0.0 to 1.0).

    Raises
    ------
    Flow360ValueError
        If no examples are provided or no match is found.
    """
    if not examples:
        raise Flow360ValueError("No examples available to search.")

    query_lower = query_name.lower().strip()
    best_match = None
    best_score = 0.0

    for example in examples:
        example_title_lower = example.title.lower()
        score = difflib.SequenceMatcher(None, query_lower, example_title_lower).ratio()

        if score > best_score:
            best_score = score
            best_match = example

    if best_match is None or best_score < 0.3:
        available_names = [ex.title for ex in examples]
        raise Flow360ValueError(
            f"No matching example found for '{query_name}'. "
            f"Available examples: {', '.join(available_names[:5])}"
            + (f" (and {len(available_names) - 5} more)" if len(available_names) > 5 else "")
        )

    return best_match, best_score


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
        log.info("No examples available.")
        return

    examples_url = Env.current.get_web_real_url("examples")
    log.info(f"These examples are pre-executed projects that can be visited on {examples_url}")
    log.info("")

    title_width = max(len(e.title) for e in examples)
    id_width = max(len(e.id) for e in examples)

    header = f"{'#':>3}  {'Title'.ljust(title_width)}  {'Example ID'.ljust(id_width)}  Tags"
    table_string = ""
    table_string += header + "\n"
    table_string += "-" * len(header) + "\n"

    for idx, ex in enumerate(examples):
        title = ex.title
        example_id = ex.id
        tags = ", ".join(ex.tags)
        table_string += (
            f"{idx+1:>3}  {title.ljust(title_width)}  {example_id.ljust(id_width)}  {tags}\n"
        )

    log.info(table_string)


def _get_project_copy_status(project_id: str) -> Optional[str]:
    """
    Get the copy status of a project.

    Parameters
    ----------
    project_id : str
        Project ID to check.

    Returns
    -------
    Optional[str]
        Copy status of the project, or None if not available.
    """
    try:
        project_api = RestApi(ProjectInterface.endpoint, id=project_id)
        info = project_api.get()
        if isinstance(info, dict):
            return info.get("copyStatus")
    except Flow360Error:
        pass
    return None


def _wait_for_copy_completion(project_id: str, timeout_minutes: int = 30) -> None:
    """
    Wait for the copy operation to complete.

    Parameters
    ----------
    project_id : str
        Project ID to monitor.
    timeout_minutes : int
        Maximum time to wait in minutes.

    Raises
    ------
    TimeoutError
        If the copy operation doesn't complete within the timeout period.
    """
    update_every_seconds = 2
    start_time = time.time()
    max_dots = 30

    with log.status() as status_logger:
        while True:
            copy_status = _get_project_copy_status(project_id)
            if copy_status != "copying":
                break

            elapsed = time.time() - start_time
            dot_count = int((elapsed // update_every_seconds) % max_dots)
            status_logger.update(f"Copying example{'.' * dot_count}")

            if time.time() - start_time > timeout_minutes * 60:
                raise TimeoutError(
                    f"Timeout: Copy operation did not finish within {timeout_minutes} minutes."
                )

            time.sleep(update_every_seconds)


def copy_example(example_id: str, wait_for_completion: bool = True) -> str:
    """
    Copy an example from the cloud and return the created project ID.

    Parameters
    ----------
    example_id : str
        ID of the example to copy.
    wait_for_completion : bool
        Whether to wait for the copy operation to complete before returning.
        Default is True (blocking).

    Returns
    -------
    str
        Project ID of the newly created project.

    Raises
    ------
    Flow360WebError
        If the example cannot be copied or the response format is unexpected.
    TimeoutError
        If wait_for_completion is True and the copy doesn't finish within timeout.
    """
    request = CopyExampleRequest(source_example_id=example_id)
    example_api = RestApi("v2/examples")
    resp = example_api.post(request.dict(), method="copy")
    if not isinstance(resp, dict):
        raise Flow360WebError(f"Unexpected response format when copying example {example_id}")
    response_model = CopyExampleResponse(**resp)
    project_id = response_model.id

    if wait_for_completion:
        copy_status = _get_project_copy_status(project_id)
        if copy_status == "copying":
            log.info(f"Copy operation started for project {project_id}. Waiting for completion...")
            _wait_for_copy_completion(project_id)
            log.info("Copy operation completed successfully.")

    return project_id
