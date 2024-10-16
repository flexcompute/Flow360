
from typing import List, Tuple, Any, Optional, Union, Literal
import pydantic as pd
from pydantic import PrivateAttr
import backoff
import aiohttp
import asyncio
import os
import json
from functools import wraps

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.log import log
from flow360.exceptions import Flow360WebError, Flow360WebNotFoundError

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
                raise Flow360WebNotAvailableError(f"Web {args[1]}: not available {error_message}")

            if resp.status == 400:
                error_message = await resp.json()
                log.debug(f"{error_message=}")
                raise Flow360WebError(
                    f"Web {args=}, {kwargs=}: Bad request error: {error_message}"
                )

            if resp.status == 404:
                error_message = await resp.json()
                raise Flow360WebNotFoundError(f"Web {args[1]}: Not found error: {error_message}")

            if resp.status == 200:
                log.debug('I am here!!!!!')
                try:
                    content_type = resp.headers.get('Content-Type', '')
                    log.debug(f'{content_type=}')
                    if 'application/json' in content_type:
                        result = await resp.json()
                        log.debug(f'Response received: {result}')
                        return result.get("data")
                    else:
                        # Handle binary content like ZIP files
                        log.debug('handling binary content!!!!!!!!!')
                        data = await resp.read()
                        log.debug(f"Binary content received. {len(data)=}")
                        # zip_file_path = "images.zip"
                        # with open(zip_file_path, 'wb') as f:
                        #     f.write(await resp.read())
                        #     log.info(f"Zip file saved to {zip_file_path}")
                        return {"resp": resp, "data": data} # Return the raw binary data
                except Exception as e:
                    log.error(f"Exception occurred while reading response: {e}")
                    raise

            error_message = await resp.text()
            raise Flow360WebError(f"Web {args[1]}: Unexpected response error: {resp.status}, {error_message}")

    return wrapper


class UVFshutter(Flow360BaseModel):
    cases: List[Any]
    data_storage: str = "."
    _url: str = PrivateAttr("https://uvf-shutter.dev-simulation.cloud")


    async def _get_3d_images(
        self, screenshots: List[Tuple]
    ) -> list[str]:
        # @backoff.on_exception(backoff.expo, ValueError, max_time=300)
        @http_interceptor
        async def _get_image_sequence(session: aiohttp.client.ClientSession, url: str, uvf_request: list[dict]) -> str:
            log.debug(f'sending request to uvf-shutter: {url=}, {type(uvf_request)=}, {len(uvf_request)=}')
            return session.post(url, json=uvf_request)
            async with session.post(url, json=uvf_request) as response:
                # zip_file_path = "images.zip"

                # if response.status == 200:
                #     with open(zip_file_path, 'wb') as f:
                #         f.write(await response.read())
                #     log.info(f"Zip file saved to {zip_file_path}")
                return response 


        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            tasks = []
            img_names = []
            for i, (img_name, uvf_request) in enumerate(screenshots):
                img_names.append(img_name)
                # task = _get_image_screeshot(session=session, url=self._url + "/screenshot", uvf_request=uvf_request)

                task = _get_image_sequence(session=session, url=self._url + "/sequence/run", uvf_request=uvf_request[i])

                tasks.append(task)
                
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            # log.debug(f"these are my responses: {responses}")

            zip_file_path = "images.zip"
            for response in responses:
                if isinstance(response, )
                    log.debug(f"loop: this is my responses: {response=}, {type(response)=}")
                except:
                    pass

                # if response.status == 200:
                with open(zip_file_path, 'wb') as f:
                    f.write(response)
                log.info(f"Zip file saved to {zip_file_path}")
                # else:
                #     raise ValueError(f"Error: {response.status}")



            import zipfile
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall("extracted_images")
                log.info(f"Extracted files: {zip_ref.namelist()}")
            
            # List the files in the extracted folder
            extracted_files = os.listdir("extracted_images")
            print(f"Extracted files: {extracted_files}")

            # for response, img_name in zip(responses, img_names):
            #     with open(img_name, "wb") as img_out:
            #         img_out.write(response)

        return img_names

    def get_images(self, fig_name_prefix, data, use_mock_manifest: bool=False):
        screenshots = []
        img_names = []

        for case in self.cases:
            img_name = os.path.join(self.data_storage, fig_name_prefix + "_" + case.name + ".png")
            img_names.append(img_name)
            if not os.path.exists(img_name):
                screenshots.append((img_name, data))
            else:
                log.debug(f'File: {img_name=} exists in cache, reusing.')

        asyncio.run(
            self._get_3d_images(screenshots)
        )

        return img_names