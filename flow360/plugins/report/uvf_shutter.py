
from typing import List, Tuple, Any, Optional, Union, Literal
import pydantic as pd
from pydantic import PrivateAttr
import backoff
import aiohttp
import asyncio
import os
import json

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.log import log

here = os.path.dirname(os.path.abspath(__file__))



class UVFShutterRequestBaseModel(Flow360BaseModel):
    def model_dump_json(self, **kwargs):
        return super().model_dump_json(by_alias=True, **kwargs)


OBJECT_TYPES = Literal[
    "slices", "qcriterion", "boundaries"
]

class Resolution(Flow360BaseModel):
    width: int = 3840
    height: int = 2160

class Settings(Flow360BaseModel):
    resolution: Resolution = Resolution()

class SetObjectVisibilityPayload(Flow360BaseModel):
    object_ids: List[OBJECT_TYPES]
    visibility: bool

class SetFieldPayload(Flow360BaseModel):
    object_id: OBJECT_TYPES
    field_name: str
    min_max: Tuple[float, float]

class TakeScreenshotPayload(Flow360BaseModel):
    file_name: str = pd.Field(alias='filename')
    type: str = 'png'

class ResetFieldPayload(Flow360BaseModel):
    object_id: OBJECT_TYPES

ACTION_TYPES = Literal[
    "focus",
    "set-object-visibility",
    "set-field",
    "reset-field",
    "take-screenshot"
]

class ActionPayload(Flow360BaseModel):
    action: ACTION_TYPES
    payload: Optional[
        Union[
            SetObjectVisibilityPayload, 
            SetFieldPayload, 
            TakeScreenshotPayload, 
            ResetFieldPayload
        ]
    ] = None

class Scene(Flow360BaseModel):
    name: str
    settings: Settings = Settings()
    script: List[ActionPayload]

class ScenesData(Flow360BaseModel):
    scenes: List[Scene]



class UVFshutter(Flow360BaseModel):
    cases: List[Any]
    data_storage: str = "."
    _url: str = PrivateAttr("https://uvf-shutter.dev-simulation.cloud")


    async def _get_3d_images(
        self, screenshots: List[Tuple]
    ) -> list[str]:
        @backoff.on_exception(backoff.expo, ValueError, max_time=300)
        async def _get_image_screeshot(session: aiohttp.client.ClientSession, url: str, manifest: list[dict]):
            log.debug(f'sending request to uvf-shutter: {url=}, {type(manifest)=}, {len(manifest)=}')
            async with session.post(url, json=manifest) as response:
                if response.status == 503:
                    raise ValueError("503 response received.")
                else:
                    return await response.read()

        async def _get_image_sequence(session: aiohttp.client.ClientSession, url: str, manifest: list[dict]):
            log.debug(f'sending request to uvf-shutter: {url=}, {type(manifest)=}, {len(manifest)=}')
            async with session.post(url, json=manifest) as response:
                if response.status == 503:
                    raise ValueError("503 response received.")
                else:
                    return await response.read()


        async with aiohttp.ClientSession() as session:
            tasks = []
            img_names = []
            for img_name, manifest in screenshots:
                img_names.append(img_name)
                # task = _get_image_screeshot(session=session, url=self._url + "/screenshot", manifest=manifest)

                task = _get_image_sequence(session=session, url=self._url + "//sequence/run", manifest=manifest)

                tasks.append(task)
            responses = await asyncio.gather(*tasks)

            for response, img_name in zip(responses, img_names):
                with open(img_name, "wb") as img_out:
                    img_out.write(response)

        return img_names

    def get_images(self, fig_name_prefix, use_mock_manifest: bool=False):
        screenshots = []
        img_names = []

        for case in self.cases:
            img_name = os.path.join(self.data_storage, fig_name_prefix + "_" + case.name + ".png")
            img_names.append(img_name)
            if not os.path.exists(img_name):
                if use_mock_manifest is True:
                    with open(os.path.join(here, "mock_manifest.json"), "r") as in_file:
                        manifest = json.load(in_file)
                    screenshots.append((img_name, manifest))
                else:    
                    screenshots.append((img_name, case._get_manifest()))
            else:
                log.debug(f'File: {img_name=} exists in cache, reusing.')

        asyncio.run(
            self._get_3d_images(screenshots)
        )

        return img_names