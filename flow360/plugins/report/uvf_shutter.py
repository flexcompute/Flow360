"""
This module is reponsible for communicating with UVF-shutter service
"""

import asyncio
import json
import os
import reprlib
import shutil
import subprocess
import time
import zipfile
from collections import defaultdict
from functools import wraps
from typing import Any, List, Literal, Optional, Tuple, Union, Callable
from urllib.parse import urljoin

# this plugin is optional, thus pylatex is not required: TODO add handling of installation of aiohttp, backoff
# pylint: disable=import-error
import aiohttp
import backoff
import pydantic as pd

from flow360 import Env
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.exceptions import Flow360WebError, Flow360WebNotFoundError
from flow360.log import log

here = os.path.dirname(os.path.abspath(__file__))


class ShutterRequestBaseModel(Flow360BaseModel):
    """
    Base model for UVF shutter requests
    """

    # def model_dump_json(self, **kwargs):
    #     return super().model_dump_json(by_alias=True, **kwargs)


ShutterObjectTypes = Literal["slices", "qcriterion", "isosurfaces", "boundaries", "edges"]


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
    width : int, default=3840
        Width of the resolution in pixels.
    height : int, default=2160
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
    file_name: str = pd.Field(alias="filename")
    type: str = "png"
    width: Optional[pd.PositiveInt] = 400
    height: Optional[pd.PositiveInt] = 60
    title: Optional[str] = None  # use to overwrite title, default is field name


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

    object_id: Union[ShutterObjectTypes, str]
    visibility: bool


class Camera(Flow360BaseModel):
    """
    Represents the camera configuration payload.

    Attributes
    ----------
    position : Vector3
        Camera eye position, think of the eye position as a position on the unit sphere centered at the `lookAt`.
    up : Vector3
        Up orientation of the camera.
    look_at : Vector3
        Target point the camera will look at from the position. Default: center of bbox
    pan_target : Vector3 or None
        Position to pan the viewport center to; if undefined, the default is `look_at`.
    dimension_direction : {'width', 'height', 'diagonal'}
        The direction `dimension_size_model_units` is for.
    dimension_size_model_units : float
        The camera zoom will be set such that the extents of the scene's projection is this number of model units for the applicable `dimensionDirection`.
    """

    position: Optional[Tuple[float, float, float]] = (-1, -1, 1)
    up: Optional[Tuple[float, float, float]] = (0, 0, 1)
    look_at: Optional[Tuple[float, float, float]] = None
    pan_target: Optional[Tuple[float, float, float]] = None
    dimension_dir: Optional[Literal["width", "height", "diagonal"]] = pd.Field(
        "width", alias="dimensionDirection"
    )
    dimension: Optional[float] = pd.Field(None, alias="dimensionSizeModelUnits")


class SetCameraPayload(Camera):
    pass


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
    if eq(model_a, model_b):
        attr_a = getattr(model_a, key_to_combine)
        attr_b = getattr(model_b, key_to_combine)

        if isinstance(attr_a, list) and isinstance(attr_b, list):
            combined_attr = attr_a + attr_b
            new_model = model_a.copy(update={key_to_combine: combined_attr})
            return new_model, True
        else:
            raise TypeError(f"The attribute '{key_to_combine}' is not a list in both models.")

    return model_a, False


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
        - If a 'set-field' action has the same payload as the previous one and no intervening actions affect the field settings,
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
                else:
                    last_set_field_payload = payload
            elif action.action in field_affecting_actions:
                last_set_field_payload = None
            optimized_actions.append(action)

        return optimized_actions

    def add_request(self, request: ScenesData):
        stored_scenes = self._merge_similar_scenes(request.resource, request.scenes)

        for stored_scene in stored_scenes:
            stored_scene.script = self._merge_visibility_actions(stored_scene.script)

        for stored_scene in stored_scenes:
            stored_scene.script = self._remove_redundant_visibility_actions(stored_scene.script)

        for stored_scene in stored_scenes:
            stored_scene.script = self._remove_redundant_set_field_actions(stored_scene.script)

        self.requests[request.resource] = stored_scenes

    def get_batch_requests(self):
        return [
            ScenesData(resource=resource, scenes=scenes)
            for resource, scenes in self.requests.items()
        ]

    def process_requests(self, context):
        """
        Processes the collected requests by grouping and combining them.
        """

        context_data = {
            "data_storage": context.data_storage,
            "url": context.shutter_url,
            "access_token": context.shutter_access_token,
            "screeshot_process_function": context.shutter_screeshot_process_function,
        }
        context_data = {k: v for k, v in context_data.items() if v is not None}
        img_files = Shutter(**context_data).get_images("None", self.get_batch_requests())
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
    """

    data_storage: str = "."
    url: str = pd.Field(default_factory=lambda: f"https://shutter-api.{Env.current.domain}")
    use_cache: bool = True
    access_token: Optional[str] = None
    screeshot_process_function: Optional[Callable] = None

    async def _get_3d_images_api(self, screenshots: dict[str, Tuple]) -> dict[str, list]:
        @backoff.on_exception(
            lambda: backoff.constant(3), Flow360WebNotAvailableError, max_time=3600
        )
        @http_interceptor
        async def _get_image_sequence(
            session: aiohttp.client.ClientSession, url: str, shutter_request: list[dict]
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

    def get_images(self, fig_name, data: List[ScenesData]) -> dict[str, List]:
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
                screenshots.append(
                    (
                        case_id,
                        img_folder,
                        data_item.model_dump(by_alias=True, exclude_unset=True, exclude_none=True),
                    )
                )
            else:
                log.debug(f"File: {img_name=} exists in cache, reusing.")
                if case_id not in cached_files:
                    cached_files[case_id] = [img_full_path]
                else:
                    cached_files[case_id].append(img_full_path)

        if self.screeshot_process_function is not None:
            process_function = self.screeshot_process_function
        else:
            process_function = self._get_3d_images_api
        img_files_generated = asyncio.run(process_function(screenshots))

        return {**img_files_generated, **cached_files}
