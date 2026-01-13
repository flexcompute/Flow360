"""
This module is responsible for communicating with UVF-shutter service
"""

import asyncio
import json
import os
import reprlib
import time
import zipfile
from collections import defaultdict
from functools import wraps
from typing import Callable, List, Literal, Optional, Tuple, Union
from urllib.parse import urljoin

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.environment import Env
from flow360.exceptions import (
    Flow360RuntimeError,
    Flow360WebError,
    Flow360WebNotFoundError,
)
from flow360.log import log
from flow360.plugins.report.report_context import ReportContext

here = os.path.dirname(os.path.abspath(__file__))


class ShutterRequestBaseModel(Flow360BaseModel):
    """
    Base model for UVF shutter requests
    """


ShutterObjectTypes = Literal["slices", "qcriterion", "isosurface", "boundaries"]


class Resource(Flow360BaseModel):
    """
    Resource context for identifying the source of a request.

    Parameters
    ----------
    path_prefix : str
        Prefix with identifier for the user initiating the request.
    id : str
        Identifier for the case associated with the request.
    """

    path_prefix: str
    id: str
    type: str = "case"

    model_config = Flow360BaseModel.model_config.copy()
    model_config.update({"frozen": True})


class Resolution(Flow360BaseModel):
    """
    Screen resolution settings.

    Parameters
    ----------
    width : int, default=1920
        Width of the resolution in pixels.
    height : int, default=1080
        Height of the resolution in pixels.
    """

    width: pd.PositiveInt = 1920
    height: pd.PositiveInt = 1080


class Settings(Flow360BaseModel):
    """
    Configuration settings for a scene.

    Parameters
    ----------
    resolution : Resolution
        Resolution settings for the scene.
    """

    resolution: Resolution = Resolution()

    model_config = Flow360BaseModel.model_config.copy()
    model_config.update({"frozen": True})


class SetObjectVisibilityPayload(Flow360BaseModel):
    """
    Payload for setting the visibility of objects.

    Parameters
    ----------
    object_ids : List[UvfObjectTypes]
        List of object identifiers for which visibility will be set.
    visibility : bool
        Boolean indicating the visibility state.
    """

    object_ids: List[Union[ShutterObjectTypes, str]]
    visibility: bool


class SetColormapPayload(Flow360BaseModel):
    """
    SetColormapPayload
    """

    type: Literal["Rainbow"] = "Rainbow"
    steps: Optional[
        List[float]
    ]  # eg: [0, 0.25, 0.5, 0.75, 1.0], // Always sorted before use, 0 and 1 implicitly added if not specified


class SetFieldPayload(Flow360BaseModel):
    """
    Payload for setting field parameters on an object.

    Parameters
    ----------
    object_id : ShutterObjectTypes
        Identifier of the object for field setting.
    field_name : str
        Name of the field to modify.
    min_max : Tuple[float, float]
        Minimum and maximum values for the field.
    """

    object_id: ShutterObjectTypes
    field_name: str
    min_max: Tuple[float, float]
    is_log_scale: bool
    # colormap: Optional[SetColormapPayload] = None


class TakeLegendScreenshotPayload(Flow360BaseModel):
    """
    Payload for taking a legend screenshot.

    This payload is used to capture just the legend (color scale, range, and optional title) for
    the specified object (e.g., slice, boundary). The legend is saved as an image file.

    Parameters
    ----------
    object_id : ShutterObjectTypes
        Identifier of the object whose legend will be captured.
    file_name : str
        The name of the output screenshot file, aliased as "filename".
    type : str, default="png"
        Image format for the legend screenshot.
    width : int, default=400
        Width of the screenshot in pixels.
    height : int, default=60
        Height of the screenshot in pixels.
    title : str, optional
        Title to display on the legend. If omitted, the default is the field name being visualized.
    """

    object_id: ShutterObjectTypes
    file_name: str = pd.Field(alias="filename")
    type: str = "png"
    width: Optional[pd.PositiveInt] = 400
    height: Optional[pd.PositiveInt] = 60
    title: Optional[str] = None


class SetLICPayload(Flow360BaseModel):
    """
    Payload for setting the visibility of objects.

    Parameters
    ----------
    object_id : ShutterObjectTypes
        Object identifier for which LIC will be set.
    visibility : bool
        Boolean indicating the visibility state.
    """

    object_id: str
    visibility: bool


class Camera(Flow360BaseModel):
    """
    Represents the camera configuration payload.
    """

    position: Optional[Tuple[float, float, float]] = pd.Field(
        (-1, -1, 1),
        description="Camera eye position, think of the eye position as a position on the unit sphere"
        + " centered at the `lookAt`. The units are in length units used in geometry or volume mesh.",
    )
    up: Optional[Tuple[float, float, float]] = pd.Field(
        (0, 0, 1), description="Up orientation of the camera."
    )
    look_at: Optional[Tuple[float, float, float]] = pd.Field(
        None,
        description="Target point the camera will look at from the position. Default: center of bbox."
        + " The units are in length units used in geometry or volume mesh.",
    )
    pan_target: Optional[Tuple[float, float, float]] = pd.Field(
        None,
        description="Position to pan the viewport center to; if undefined, the default is `look_at`."
        + " The units are in length units used in geometry or volume mesh.",
    )
    dimension_dir: Optional[Literal["width", "height", "diagonal"]] = pd.Field(
        "width",
        alias="dimensionDirection",
        description="The direction `dimension_size_model_units` is for.",
    )
    dimension: Optional[float] = pd.Field(
        None,
        alias="dimensionSizeModelUnits",
        description="The camera zoom will be set such that the extents of the scene's projection is this number"
        + " of model units for the applicable `dimension_dir`."
        + " The units are in length units used in geometry or volume mesh.",
    )
    type: Literal["Camera"] = pd.Field("Camera", frozen=True)


class TopCamera(Camera):
    """
    Camera looking down from above (along +Z).
    """

    position: Tuple[float, float, float] = pd.Field((0.0, 0.0, 1.0))
    look_at: Tuple[float, float, float] = pd.Field((0.0, 0.0, 0.0))
    up: Tuple[float, float, float] = pd.Field((0.0, 1.0, 0.0))
    type: Literal["TopCamera"] = pd.Field("TopCamera")


class LeftCamera(Camera):
    """
    Camera looking from the positive Y side toward the origin (i.e. along -Y).
    """

    position: Tuple[float, float, float] = pd.Field((0.0, -1.0, 0.0))
    look_at: Tuple[float, float, float] = pd.Field((0.0, 0.0, 0.0))
    up: Tuple[float, float, float] = pd.Field((0.0, 0.0, 1.0))
    type: Literal["LeftCamera"] = pd.Field("LeftCamera")


class RearCamera(Camera):
    """
    Camera looking from negative X toward the origin (i.e. along +X).
    """

    position: Tuple[float, float, float] = pd.Field((1.0, 0.0, 0.0))
    look_at: Tuple[float, float, float] = pd.Field((0.0, 0.0, 0.0))
    up: Tuple[float, float, float] = pd.Field((0.0, 0.0, 1.0))
    type: Literal["RearCamera"] = pd.Field("RearCamera")


class FrontCamera(Camera):
    """
    Camera looking from positive X side toward the origin (i.e. along -X).
    """

    position: Tuple[float, float, float] = pd.Field((-1.0, 0.0, 0.0))
    look_at: Tuple[float, float, float] = pd.Field((0.0, 0.0, 0.0))
    up: Tuple[float, float, float] = pd.Field((0.0, 0.0, 1.0))
    type: Literal["FrontCamera"] = pd.Field("FrontCamera")


class BottomCamera(Camera):
    """
    Camera looking up from below (along -Z).
    """

    position: Tuple[float, float, float] = pd.Field((0.0, 0.0, -1.0))
    look_at: Tuple[float, float, float] = pd.Field((0.0, 0.0, 0.0))
    up: Tuple[float, float, float] = pd.Field((0.0, -1.0, 0.0))
    type: Literal["BottomCamera"] = pd.Field("BottomCamera")


class FrontLeftBottomCamera(Camera):
    """
    Camera placed front-left-bottom, diagonally looking at the model.]
    """

    position: Tuple[float, float, float] = pd.Field((-1.0, -1.0, -1.0))
    look_at: Tuple[float, float, float] = pd.Field((0.0, 0.0, 0.0))
    up: Tuple[float, float, float] = pd.Field((0.0, 0.0, 1.0))
    type: Literal["FrontLeftBottomCamera"] = pd.Field("FrontLeftBottomCamera")


class RearRightBottomCamera(Camera):
    """
    Camera placed rear-right-bottom, diagonally looking at the model.
    """

    position: Tuple[float, float, float] = pd.Field((1.0, 1.0, -1.0))
    look_at: Tuple[float, float, float] = pd.Field((0.0, 0.0, 0.0))
    up: Tuple[float, float, float] = pd.Field((0.0, 0.0, 1.0))
    type: Literal["RearRightBottomCamera"] = pd.Field("RearRightBottomCamera")


class FrontLeftTopCamera(Camera):
    """
    Camera placed front-left-top, diagonally looking at the model.
    """

    position: Tuple[float, float, float] = pd.Field((-1.0, -1.0, 1.0))
    look_at: Tuple[float, float, float] = pd.Field((0.0, 0.0, 0.0))
    up: Tuple[float, float, float] = pd.Field((0.0, 0.0, 1.0))
    type: Literal["FrontLeftTopCamera"] = pd.Field("FrontLeftTopCamera")


class RearLeftTopCamera(Camera):
    """
    Camera placed rear-left-top, diagonally looking at the model.
    """

    position: Tuple[float, float, float] = pd.Field((1.0, -1.0, 1.0))
    look_at: Tuple[float, float, float] = pd.Field((0.0, 0.0, 0.0))
    up: Tuple[float, float, float] = pd.Field((0.0, 0.0, 1.0))
    type: Literal["RearLeftTopCamera"] = pd.Field("RearLeftTopCamera")


class SetCameraPayload(Camera):
    """
    Alias for Camera
    """


class TakeScreenshotPayload(Flow360BaseModel):
    """
    Payload for taking a screenshot.

    Parameters
    ----------
    file_name : str
        Name of the file for saving the screenshot.
    type : str, default="png"
        Type of the screenshot file format.
    """

    file_name: str = pd.Field(alias="filename")
    type: str = "png"


class ResetFieldPayload(Flow360BaseModel):
    """
    Payload for resetting a field on an object.

    Parameters
    ----------
    object_id : ShutterObjectTypes
        Identifier of the object for which the field is reset.
    """

    object_id: ShutterObjectTypes


class FocusPayload(Flow360BaseModel):
    """
    Payload for focusing camera on an object.

    Parameters
    ----------
    object_id : ShutterObjectTypes
        Identifier of the object for which the field is reset.
    zoom: pd.PositiveFloat
        Zoom multiplier can be used to add padding, default 1
    """

    object_ids: List[Union[ShutterObjectTypes, str]]
    zoom: Optional[pd.PositiveFloat] = 1


ActionTypes = Literal[
    "focus",
    "set-object-visibility",
    "set-field",
    "set-lic",
    "reset-field",
    "set-camera",
    "take-screenshot",
    "take-legend-screenshot",
]


class ActionPayload(Flow360BaseModel):
    """
    Defines an action to be taken on an object in a scene.

    Parameters
    ----------
    action : ACTIONTYPES
        Type of action to be performed.
    payload : Optional[Union[SetObjectVisibilityPayload, SetFieldPayload, TakeScreenshotPayload, ResetFieldPayload]]
        Data required for the specified action.
    """

    action: ActionTypes
    payload: Optional[
        Union[
            SetObjectVisibilityPayload,
            SetFieldPayload,
            TakeScreenshotPayload,
            TakeLegendScreenshotPayload,
            ResetFieldPayload,
            SetCameraPayload,
            FocusPayload,
            SetLICPayload,
        ]
    ] = None


class Scene(Flow360BaseModel):
    """
    Represents a scene with specific settings and scripted actions.

    Parameters
    ----------
    name : str
        Name of the scene.
    settings : Settings
        Configuration settings for the scene.
    script : List[ActionPayload]
        List of actions to execute within the scene.
    """

    name: str
    settings: Settings = Settings()
    script: List[ActionPayload]


class ScenesData(Flow360BaseModel):
    """
    Data structure holding multiple scenes and associated resource.

    Parameters
    ----------
    resource : Resource
        Resource data related to the source.
    scenes : List[Scene]
        List of scenes with actions and settings.
    """

    resource: Resource
    scenes: List[Scene]


class Flow360WebNotAvailableError(Flow360WebError):
    """
    Exception raised when the Flow360 web service is unavailable.
    """


def http_interceptor(func):
    """
    Decorator that intercepts HTTP responses and raises appropriate exceptions based on response status.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        """A wrapper function"""

        log.debug(f"call: {func.__name__}({reprlib.repr(args)}, {reprlib.repr(kwargs)})")
        start_time = time.time()

        async with await func(*args, **kwargs) as resp:
            log.debug(f"response: {resp}")

            if resp.status == 503:
                error_message = await resp.json()
                raise Flow360WebNotAvailableError(f"Web: not available {error_message}")

            if resp.status == 400:
                error_message = await resp.json()
                log.debug(f"{error_message=}")
                raise Flow360WebError(f"Web: Bad request error: {error_message}")

            if resp.status == 404:
                error_message = await resp.json()
                raise Flow360WebNotFoundError(f"Web: Not found error: {error_message}")

            if resp.status == 200:
                end_time = time.time()
                execution_time = end_time - start_time
                log.debug(f"UVF Shutter execution time: {execution_time:.4f} seconds")

                try:
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        result = await resp.json()
                        return result.get("data")
                    data = await resp.read()
                    log.debug("received binary data")
                    return data
                except Exception as e:
                    log.error(f"Exception occurred while reading response: {e}")
                    raise

            error_message = await resp.text()
            raise Flow360WebError(f"Web: Unexpected response error: {resp.status}, {error_message}")

    return wrapper


def combine(model_a, model_b, key_to_combine, eq: callable):
    """
    Combines attributes of two models based on a specified key if they satisfy a given equality condition.

    This function checks if two models are considered equal using the provided `eq` function.
    If they are equal, it combines the values of the specified attribute (referenced by `key_to_combine`)
    into a single list, updates the first model with the combined values, and returns the updated model
    along with a flag indicating that the models were combined.

    Parameters
    ----------
    model_a : Any
        The first model to combine.
    model_b : Any
        The second model to combine.
    key_to_combine : str
        The name of the attribute in both models to combine.
    eq : callable
        A function that takes two models as input and returns `True` if the models are considered equal,
        `False` otherwise.

    Returns
    -------
    tuple
        A tuple containing:
        - The updated model (`model_a`) if the models were combined, or the original `model_a` if not.
        - A boolean flag indicating whether the models were combined (`True`) or not (`False`).

    Raises
    ------
    TypeError
        If the specified attribute (`key_to_combine`) is not a list in both models.

    Example
    -------
    >>> model_a = SomeModel(attr=["item1"])
    >>> model_b = SomeModel(attr=["item2"])
    >>> eq = lambda a, b: a.some_property == b.some_property
    >>> combined_model, combined = combine(model_a, model_b, "attr", eq)
    >>> print(combined_model.attr)  # ['item1', 'item2']
    >>> print(combined)  # True
    """

    if eq(model_a, model_b):
        attr_a = getattr(model_a, key_to_combine)
        attr_b = getattr(model_b, key_to_combine)

        if isinstance(attr_a, list) and isinstance(attr_b, list):
            combined_attr = attr_a + attr_b
            new_model = model_a.copy(update={key_to_combine: combined_attr})
            return new_model, True
        raise TypeError(f"The attribute '{key_to_combine}' is not a list in both models.")

    return model_a, False


def make_shutter_context(context: ReportContext):
    """
    Extracts relevant data for shutter from context
    """
    context_data = {
        "data_storage": context.data_storage,
        "url": context.shutter_url,
        "access_token": context.shutter_access_token,
        "screenshot_process_function": context.shutter_screenshot_process_function,
        "process_screenshot_in_parallel": context.process_screenshot_in_parallel,
    }
    context_data = {k: v for k, v in context_data.items() if v is not None}
    return context_data


class ShutterBatchService:
    """
    Service to collect and process UVF shutter requests.
    """

    def __init__(self, data_storage=".", shutter_url=None, shutter_access_token=None):
        self.data_storage = data_storage
        self.shutter_url = shutter_url
        self.shutter_access_token = shutter_access_token
        self.requests = defaultdict(list)

    def _merge_similar_scenes(self, resource, new_scenes):
        stored_scenes: List[Scene] = self.requests.get(resource, [])
        for scene in new_scenes:
            appended = False
            # pylint: disable=unnecessary-lambda-assignment
            eq = lambda a, b: a.name == b.name and a.settings == b.settings
            for i, stored_scene in enumerate(stored_scenes):
                combined_scene, appended = combine(stored_scene, scene, "script", eq)
                if appended:
                    stored_scenes[i] = combined_scene
                    break
            if appended is False:
                stored_scenes.append(scene.copy())
        return stored_scenes

    def _merge_visibility_actions(self, actions: List[ActionPayload]) -> List[ActionPayload]:
        """
        Merges consecutive 'set-object-visibility' actions with the same 'visibility' value.
        Does not merge actions if there are other actions (like 'screenshot') in between.
        """
        merged_actions = []
        last_visibility_action = None

        for action in actions:
            if action.action == "set-object-visibility":
                payload = action.payload
                if not isinstance(payload, SetObjectVisibilityPayload):
                    raise TypeError("Payload must be of type SetObjectVisibilityPayload")

                if (
                    last_visibility_action
                    and payload.visibility == last_visibility_action.payload.visibility
                ):
                    combined_object_ids = set(last_visibility_action.payload.object_ids) | set(
                        payload.object_ids
                    )
                    last_visibility_action.payload.object_ids = list(combined_object_ids)
                else:
                    if last_visibility_action:
                        merged_actions.append(last_visibility_action)
                    last_visibility_action = action.copy()
            else:
                if last_visibility_action:
                    merged_actions.append(last_visibility_action)
                    last_visibility_action = None
                merged_actions.append(action)

        if last_visibility_action:
            merged_actions.append(last_visibility_action)

        return merged_actions

    def _remove_redundant_visibility_actions(
        self, actions: List[ActionPayload]
    ) -> List[ActionPayload]:
        """
        Optimizes 'set-object-visibility' actions by removing redundant actions and sequences.

        Rules:
        - If a 'set-object-visibility' action with the same 'object_ids' and 'visibility' has already occurred,
        remove the subsequent duplicate action.
        - If a sequence of 'set-object-visibility' actions (regardless of other actions in between) is repeated
        later in the action list with the same 'object_ids' and 'visibility', remove the latter sequence.
        - Other action types are preserved and ignored in this optimization.

        Note:
        - This function assumes that the order of actions is important and preserves the order of other actions.
        """
        optimized_actions = []
        visibility_sequence = []
        current_sequence = []

        for action in actions:
            if action.action == "set-object-visibility":
                payload = action.payload
                if not isinstance(payload, SetObjectVisibilityPayload):
                    raise TypeError("Payload must be of type SetObjectVisibilityPayload")
                current_sequence.append(action)
            else:
                if current_sequence:
                    sequence_key = tuple(
                        (tuple(sorted(a.payload.object_ids)), a.payload.visibility)
                        for a in current_sequence
                    )
                    if sequence_key != visibility_sequence:
                        visibility_sequence = sequence_key
                        optimized_actions.extend(current_sequence)
                    current_sequence = []
                optimized_actions.append(action)

        if current_sequence:
            sequence_key = tuple(
                (tuple(sorted(a.payload.object_ids)), a.payload.visibility)
                for a in current_sequence
            )
            if sequence_key != visibility_sequence:
                optimized_actions.extend(current_sequence)

        return optimized_actions

    def _remove_redundant_set_field_actions(
        self, actions: List[ActionPayload]
    ) -> List[ActionPayload]:
        """
        Removes redundant 'set-field' actions when the payload has not changed between calls.

        Rules:
        - If a 'set-field' action has the same payload as the previous one and no intervening actions affect the field
        settings,
        remove the latter 'set-field' action.
        - Other action types are preserved and ignored in this optimization unless they affect the field settings.

        Note:
        - This function assumes that the order of actions is important and preserves the order of other actions.
        """
        optimized_actions = []
        last_set_field_payload = None

        field_affecting_actions = {"reset-field", "set-field"}

        for action in actions:
            if action.action == "set-field":
                payload = action.payload
                if not isinstance(payload, SetFieldPayload):
                    raise TypeError("Payload must be of type SetFieldPayload")

                if last_set_field_payload == payload:
                    continue
                last_set_field_payload = payload
            elif action.action in field_affecting_actions:
                last_set_field_payload = None
            optimized_actions.append(action)

        return optimized_actions

    def add_request(self, request: ScenesData):
        """
        Adds a new request to the batch and processes it to optimize actions.

        This method merges the provided request's scenes with any existing scenes for the same resource,
        ensuring that similar scenes are combined and redundant actions are removed.

        Parameters
        ----------
        request : ScenesData
            The new request containing scenes and associated resource information.

        Processing Steps
        ----------------
        1. Merges similar scenes based on their name and settings.
        2. Optimizes scripts within each scene by:
            - Merging consecutive 'set-object-visibility' actions with the same visibility value.
            - Removing redundant 'set-object-visibility' actions across the script.
            - Eliminating redundant 'set-field' actions when the payload is unchanged.

        Raises
        ------
        TypeError
            If an invalid payload type is encountered during processing.

        Example
        -------
        >>> resource = Resource(path_prefix="user", id="case1")
        >>> scenes = [Scene(name="scene1", settings=Settings(), script=[])]
        >>> request = ScenesData(resource=resource, scenes=scenes)
        >>> service = ShutterBatchService()
        >>> service.add_request(request)
        """

        stored_scenes = self._merge_similar_scenes(request.resource, request.scenes)

        for stored_scene in stored_scenes:
            stored_scene.script = self._merge_visibility_actions(stored_scene.script)

        for stored_scene in stored_scenes:
            stored_scene.script = self._remove_redundant_visibility_actions(stored_scene.script)

        for stored_scene in stored_scenes:
            stored_scene.script = self._remove_redundant_set_field_actions(stored_scene.script)

        self.requests[request.resource] = stored_scenes

    def get_batch_requests(self):
        """
        Retrieves all batched requests currently stored in the service.

        This method compiles the collected scenes and their associated resources into a
        list of `ScenesData` objects, which can be sent to the UVF shutter service for processing.

        Returns
        -------
        List[ScenesData]
            A list of `ScenesData` objects, where each object represents a resource and its
            associated scenes.
        """

        return [
            ScenesData(resource=resource, scenes=scenes)
            for resource, scenes in self.requests.items()
        ]

    def process_requests(self, context):
        """
        Processes the collected requests by grouping and combining them.
        """
        img_files = Shutter(**make_shutter_context(context)).get_images(
            "None", self.get_batch_requests()
        )
        return img_files


class Shutter(Flow360BaseModel):
    """
    Model representing UVF shutter request data and configuration settings.

    Parameters
    ----------
    data_storage : str, default="."
        Path to the directory where data will be stored.
    url : str
        URL endpoint for the shutter service, defaults to "https://shutter-api.{Env.current.domain}".
    use_cache : bool
        Whether to force generate data or use cached data
    process_screenshot_in_parallel : bool
        Whether to process screenshots concurrently
    """

    data_storage: str = "."
    url: str = pd.Field(default_factory=lambda: f"https://shutter-api.{Env.current.domain}")
    use_cache: bool = True
    process_screenshot_in_parallel: bool = True
    access_token: Optional[str] = None
    screenshot_process_function: Optional[Callable] = None

    # pylint: disable=too-many-locals
    async def _get_3d_images_api(self, screenshots: dict[str, Tuple]) -> dict[str, list]:
        try:
            import backoff  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise RuntimeError(
                "backoff is not installed. Please install backoff to use this functionality."
            ) from err

        try:
            import aiohttp  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise RuntimeError(
                "aiohttp is not installed. Please install aiohttp to use this functionality."
            ) from err

        @backoff.on_exception(
            lambda: backoff.constant(3), Flow360WebNotAvailableError, max_time=3600
        )
        @http_interceptor
        async def _get_image_sequence(
            session: aiohttp.ClientSession, url: str, shutter_request: list[dict]
        ) -> str:
            log.debug(
                f"sending request to uvf-shutter: {url=}, {type(shutter_request)=}, {len(json.dumps(shutter_request))=}"
            )
            return session.post(url, json=shutter_request)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600),
            headers={"Authorization": f"Bearer {self.access_token}"},
        ) as session:
            tasks = []
            for _, _, shutter_request in screenshots:
                tasks.append(
                    _get_image_sequence(
                        session=session,
                        url=urljoin(self.url, "/sequence/run"),
                        shutter_request=shutter_request,
                    )
                )
                log.debug(f"request to shutter: {json.dumps(shutter_request)}")

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for response, (_, img_folder, _) in zip(responses, screenshots):
                if not isinstance(response, Exception):
                    os.makedirs(img_folder, exist_ok=True)
                    zip_file_path = os.path.join(img_folder, "images.zip")
                    with open(zip_file_path, "wb") as f:
                        f.write(response)
                    log.info(f"Zip file saved to {zip_file_path}")

            for response in responses:
                if isinstance(response, Exception):
                    raise response

            img_files = {}
            for case_id, img_folder, _ in screenshots:
                zip_file_path = os.path.join(img_folder, "images.zip")
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(path=img_folder)
                    extracted = zip_ref.namelist()
                    img_files[case_id] = [os.path.join(img_folder, file) for file in extracted]
                    log.info(f"Extracted files: {extracted}")

        return img_files

    async def sequential_screenshot_generator(
        self, screenshots: List[Tuple], process_function: Optional[Callable]
    ):
        """
        process screenshot sequentially
        """
        img_files_generated = {}
        for screenshot in screenshots:
            img_files_generated_single_run = await process_function([screenshot])
            img_files_generated.update(img_files_generated_single_run)
        return img_files_generated

    # pylint: disable=too-many-branches
    def get_images(
        self, fig_name, data: List[ScenesData], regenerate_if_not_found: bool = True
    ) -> dict[str, List]:
        """
        Generates or retrieves cached image files for scenes.

        Parameters
        ----------
        fig_name : str
            The base name for the image file.
        data : List[ScenesData]
            A list of scene data objects to generate images for.

        Returns
        -------
        dict[str, List[str]]
            A dictionary with case IDs as keys and lists of image file paths as values.
        """
        screenshots = []
        cached_files = {}
        for data_item in data:
            case_id = data_item.resource.id
            img_folder = os.path.join(self.data_storage, case_id)
            img_name = fig_name + ".png"
            img_full_path = os.path.join(img_folder, img_name)
            if not os.path.exists(img_full_path) or self.use_cache is False:
                if regenerate_if_not_found is True:
                    screenshots.append(
                        (
                            case_id,
                            img_folder,
                            data_item.model_dump(
                                by_alias=True, exclude_unset=True, exclude_none=True
                            ),
                        )
                    )
                else:
                    raise Flow360RuntimeError(
                        f"File: {img_name=} not found, shutter generation failed."
                    )
            else:
                log.debug(f"File: {img_name=} exists in cache, using.")
                if case_id not in cached_files:
                    cached_files[case_id] = [img_full_path]
                else:
                    cached_files[case_id].append(img_full_path)

        img_files_generated = {}
        if len(screenshots) > 0:
            if self.screenshot_process_function is not None:
                process_function = self.screenshot_process_function
            else:
                process_function = self._get_3d_images_api

            if self.process_screenshot_in_parallel:
                img_files_generated = asyncio.run(process_function(screenshots))
            else:
                img_files_generated = asyncio.run(
                    self.sequential_screenshot_generator(screenshots, process_function)
                )

        return {**img_files_generated, **cached_files}
