"""
Validator API
"""
from enum import Enum
from typing import Union

from ..cloud.rest_api import RestApi
from ..exceptions import ValidationError
from ..exceptions import ValueError as FlValueError
from ..log import log
from .flow360_params.flow360_params import Flow360Params, UnvalidatedFlow360Params
from .meshing.params import SurfaceMeshingParams, VolumeMeshingParams


class Validator(Enum):
    """ ":class: Validator"""

    VOLUME_MESH = "VolumeMesh"
    SURFACE_MESH = "SurfaceMesh"
    CASE = "Case"

    def _get_url(self):
        if self is Validator.VOLUME_MESH:
            return "validator/volumemesh/validate"
        if self is Validator.SURFACE_MESH:
            return "validator/surfacemesh/validate"
        if self is Validator.CASE:
            return "validator/case/validate"

        return None

    # pylint: disable=anomalous-backslash-in-string
    def validate(
        self,
        params: Union[Flow360Params, SurfaceMeshingParams, VolumeMeshingParams],
        solver_version: str = None,
        mesh_id=None,
        raise_on_error: bool = True,
    ):
        """API validator

        Parameters
        ----------
        params : Union[Flow360Params, SurfaceMeshingParams]
            flow360 parameters to validate
        solver_version : str, optional
            solver version, by default None
        mesh_id : optional
            mesh ID associated with Case

        Returns
        -------
        None
            None if validation is correct

        Raises
        ------
        ValueError
            when parameters are not valid
        ValidationError
            when validation API fails
        """
        if (
            not isinstance(params, Flow360Params)
            and not isinstance(params, SurfaceMeshingParams)
            and not isinstance(params, VolumeMeshingParams)
            and not isinstance(params, UnvalidatedFlow360Params)
        ):
            raise FlValueError(
                f"""
                params must be instance of [Flow360Params, SurfaceMeshingParams, VolumeMeshingParams,
                but {params}, type={type(params)} got.
                """
            )

        api = RestApi(self._get_url())
        body = {"jsonConfig": params.to_flow360_json(), "version": solver_version}

        if mesh_id is not None:
            body["meshId"] = mesh_id

        try:
            res = api.post(body)
        # pylint: disable=broad-except
        except Exception:
            return None

        if "validationWarning" in res and res["validationWarning"] is not None:
            res_str = str(res["validationWarning"]).replace("[", "\[")
            log.warning(f"warning when validating: {res_str}")

        if "success" in res and res["success"] is True:
            return res

        if "success" in res and res["success"] is False:
            if "validationError" in res and res["validationError"] is not None:
                res_str = str(res).replace("[", "\[")
                if raise_on_error:
                    raise ValidationError(f"Error when validating: {res_str}")
                # pylint: disable=pointless-exception-statement
                ValidationError(f"Error when validating: {res_str}")

        return None
