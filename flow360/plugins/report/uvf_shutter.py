"""
This module is reponsible for communicating with UVF-shutter service
"""

import asyncio
import os
import time
import zipfile
from functools import wraps
from typing import Any, List, Literal, Optional, Tuple, Union
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


class UVFShutterRequestBaseModel(Flow360BaseModel):
    """
    Base model for UVF shutter requests
    """

    def model_dump_json(self, **kwargs):
        return super().model_dump_json(by_alias=True, **kwargs)


UvfObjectTypes = Literal["slices", "qcriterion", "boundaries", "edges"]


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

    width: pd.PositiveInt = 3840
    height: pd.PositiveInt = 2160


class Settings(Flow360BaseModel):
    """
    Configuration settings for a scene.

    Parameters
    ----------
    resolution : Resolution
        Resolution settings for the scene.
    """

    resolution: Resolution = Resolution()


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

    object_ids: List[Union[UvfObjectTypes, str]]
    visibility: bool


class SetFieldPayload(Flow360BaseModel):
    """
    Payload for setting field parameters on an object.

    Parameters
    ----------
    object_id : UvfObjectTypes
        Identifier of the object for field setting.
    field_name : str
        Name of the field to modify.
    min_max : Tuple[float, float]
        Minimum and maximum values for the field.
    """

    object_id: UvfObjectTypes
    field_name: str
    min_max: Tuple[float, float]
    is_log_scale: bool


class SetLICPayload(Flow360BaseModel):
    """
    Payload for setting the visibility of objects.

    Parameters
    ----------
    object_id : UvfObjectTypes
        Object identifier for which LIC will be set.
    visibility : bool
        Boolean indicating the visibility state.
    """

    object_id: Union[UvfObjectTypes, str]
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
    object_id : UvfObjectTypes
        Identifier of the object for which the field is reset.
    """

    object_id: UvfObjectTypes


class FocusPayload(Flow360BaseModel):
    """
    Payload for focusing camera on an object.

    Parameters
    ----------
    object_id : UvfObjectTypes
        Identifier of the object for which the field is reset.
    zoom: pd.PositiveFloat
        Zoom multiplier can be used to add padding, default 1
    """

    object_ids: List[Union[UvfObjectTypes, str]]
    zoom: Optional[pd.PositiveFloat] = 1


ActionTypes = Literal[
    "focus",
    "set-object-visibility",
    "set-field",
    "set-lic",
    "reset-field",
    "set-camera",
    "take-screenshot",
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

        log.debug(f"call: {func.__name__}({args}, {kwargs})")
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


class UVFshutter(Flow360BaseModel):
    """
    Model representing UVF shutter request data and configuration settings.

    Parameters
    ----------
    cases : List[Any]
        List of case objects associated with the UVF shutter.
    data_storage : str, default="."
        Path to the directory where data will be stored.
    url : str
        URL endpoint for the shutter service, defaults to "https://shutter-api-development.{Env.current.domain}".
    use_cache : bool
        Whether to force generate data or use cached data
    """

    cases: List[Any]
    data_storage: str = "."
    url: str = pd.Field(
        default_factory=lambda: f"https://shutter-api-development.{Env.current.domain}"
    )
    use_cache: bool = True
    access_token: Optional[str] = None

    async def _get_3d_images(self, screenshots: dict[str, Tuple]) -> dict[str, list]:
        @backoff.on_exception(backoff.expo, Flow360WebNotAvailableError, max_time=3600)
        @http_interceptor
        async def _get_image_sequence(
            session: aiohttp.client.ClientSession, url: str, uvf_request: list[dict]
        ) -> str:
            log.debug(
                f"sending request to uvf-shutter: {url=}, {type(uvf_request)=}, {len(uvf_request)=}"
            )
            return session.post(url, json=uvf_request, )

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600),
            headers={"Authorization": f"Bearer {self.access_token}"},
        ) as session:
            tasks = []
            for _, _, uvf_request in screenshots:
                tasks.append(
                    _get_image_sequence(
                        session=session,
                        url=urljoin(self.url, "/sequence/run"),
                        uvf_request=uvf_request,
                    )
                )

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
                    (case_id, img_folder, data_item.model_dump(by_alias=True, exclude_unset=True))
                )
            else:
                log.debug(f"File: {img_name=} exists in cache, reusing.")
                if case_id not in cached_files:
                    cached_files[case_id] = [img_full_path]
                else:
                    cached_files[case_id].append(img_full_path)

        img_files_generated = asyncio.run(self._get_3d_images(screenshots))

        return {**img_files_generated, **cached_files}
