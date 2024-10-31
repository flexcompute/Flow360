import asyncio
import json
import os
import zipfile
from functools import wraps
from typing import Any, List, Literal, Optional, Tuple, Union

import aiohttp
import backoff
import pydantic as pd
from pydantic import PrivateAttr

from flow360 import Env
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.exceptions import Flow360WebError, Flow360WebNotFoundError
from flow360.log import log

here = os.path.dirname(os.path.abspath(__file__))


class UVFShutterRequestBaseModel(Flow360BaseModel):
    def model_dump_json(self, **kwargs):
        return super().model_dump_json(by_alias=True, **kwargs)


UvfObjectTypes = Literal["slices", "qcriterion", "boundaries"]


class SourceContext(Flow360BaseModel):
    user_id: str
    case_id: str


class Resolution(Flow360BaseModel):
    width: int = 3840
    height: int = 2160


class Settings(Flow360BaseModel):
    resolution: Resolution = Resolution()


class SetObjectVisibilityPayload(Flow360BaseModel):
    object_ids: List[UvfObjectTypes]
    visibility: bool


class SetFieldPayload(Flow360BaseModel):
    object_id: UvfObjectTypes
    field_name: str
    min_max: Tuple[float, float]


class TakeScreenshotPayload(Flow360BaseModel):
    file_name: str = pd.Field(alias="filename")
    type: str = "png"


class ResetFieldPayload(Flow360BaseModel):
    object_id: UvfObjectTypes


ACTION_TYPES = Literal[
    "focus", "set-object-visibility", "set-field", "reset-field", "take-screenshot"
]


class ActionPayload(Flow360BaseModel):
    action: ACTION_TYPES
    payload: Optional[
        Union[SetObjectVisibilityPayload, SetFieldPayload, TakeScreenshotPayload, ResetFieldPayload]
    ] = None


class Scene(Flow360BaseModel):
    name: str
    settings: Settings = Settings()
    script: List[ActionPayload]


class ScenesData(Flow360BaseModel):
    context: SourceContext
    scenes: List[Scene]


class Flow360WebNotAvailableError(Flow360WebError):
    pass


def http_interceptor(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        """A wrapper function"""

        log.debug(f"call: {func.__name__}({args}, {kwargs})")
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
                try:
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        result = await resp.json()
                        return result.get("data")
                    else:
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
    cases: List[Any]
    data_storage: str = "."
    url: str = pd.Field(default_factory=lambda: f"https://shutter.{Env.current.domain}")

    async def _get_3d_images(self, screenshots: dict[str, Tuple]) -> dict[str, list]:
        @backoff.on_exception(backoff.expo, Flow360WebNotAvailableError, max_time=300)
        @http_interceptor
        async def _get_image_sequence(
            session: aiohttp.client.ClientSession, url: str, uvf_request: list[dict]
        ) -> str:
            log.debug(
                f"sending request to uvf-shutter: {url=}, {type(uvf_request)=}, {len(uvf_request)=}"
            )
            return session.post(url, json=uvf_request)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            tasks = []
            for _, _, uvf_request in screenshots:
                tasks.append(
                    _get_image_sequence(
                        session=session, url=self.url + "/sequence/run", uvf_request=uvf_request
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
            for id, img_folder, _ in screenshots:
                zip_file_path = os.path.join(img_folder, "images.zip")
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(path=img_folder)
                    extracted = zip_ref.namelist()
                    img_files[id] = [os.path.join(img_folder, file) for file in extracted]
                    log.info(f"Extracted files: {extracted}")

        return img_files

    def get_images(self, fig_name, data: List[ScenesData]) -> dict[str, List]:
        screenshots = []
        cached_files = {}
        for data_item in data:
            id = data_item.context.case_id
            img_folder = os.path.join(self.data_storage, id)
            img_name = fig_name + ".png"
            img_full_path = os.path.join(img_folder, img_name)
            if not os.path.exists(img_full_path):
                screenshots.append(
                    (id, img_folder, data_item.model_dump(by_alias=True, exclude_unset=True))
                )
            else:
                log.debug(f"File: {img_name=} exists in cache, reusing.")
                if id not in cached_files:
                    cached_files[id] = [img_full_path]
                else:
                    cached_files[id].append(img_full_path)

        img_files_generated = asyncio.run(self._get_3d_images(screenshots))

        return {**img_files_generated, **cached_files}
