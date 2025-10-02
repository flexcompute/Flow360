from typing import Optional, Union, Tuple, Literal
from pydantic import Field, model_validator, PrivateAttr


#CHEKIAR

# results.CFL????

# fundamentalmente eso


class createCFLplot(BaseChart2D):
    """
    Residuals is an object for showing the solution history of the CFL.
    """
    # pseudo for steady physical for unsteady

    isUnsteady: Optional[bool] = Field(
        False, description ="Specify whether the simulation is steady or unsteady"
    )
    show_grid: Optional[bool] = Field(
        True, description="If ``True``, grid lines are displayed on the plot. Defaults to ``True``."
    )
    separate_plots: Optional[bool] = Field(
        True, description="If ``True``, each residual component is plotted in a separate subplot."
    )
    xlim: Optional[Union[ManualLimit, Tuple[float, float]]] = Field(
        None,
        description="Limits for the *x*-axis. Can be a tuple ``(xmin, xmax)`` or a `ManualLimit`.",
    )

    section_title: Literal["CFL"] = Field("CFL", frozen=True)

    # private/internal attribute (not visible to users)
    _x: str = PrivateAttr(default=None)

    @model_validator(mode="after")  # checkiar como se llama cfl dentro de results
    def set_x(self):
        self._x = (
            "CFL/physical_step"
            if self.isUnsteady
            else "CFL/pseudo_step"
        )
        return self
    

    y_log: Literal[True] = Field(True, frozen=True)
    
    _requirements: List[RequirementItem] = [
        RequirementItem.from_data_key(data_key="CFL")  # esto no estoy 100% seguro q sea asi
    ]
    # Internal tag 
    type_name: Literal["CFLplot"] = Field("CFLplot", frozen=True)

    def get_requirements(self):
        """
        Returns requirements for this item.
        """
        return self._requirements

    def _get_background_chart(self, _):
        return None
    #IMPORTANT (this should work check with Piotr if it doesnt)
    def _handle_legend(self, cases, _, y_data):
        cols_exclude = cases[0].results.CFL.x_columns
        legend = []
        for case in cases:
            y_variables = [
                f"linear_residuals/{res}"
                for res in case.results.linear_residuals.as_dict().keys()
                if res not in cols_exclude
            ]

            legend += [
                (
                    f"{case.name} - {path_variable_name(y)}"
                    if len(cases) > 1
                    else f"{path_variable_name(y)}"
                )
                for y in y_variables
            ]

        return legend
    
    # Check if this section actually does anythign (doubt it)
    def _handle_secondary_x_axis(self, cases, x_data, x_lim, x_label):
        secondary_x_data, seondary_x_label = super()._handle_secondary_x_axis(
            cases, x_data, x_lim, x_label
        )
        if secondary_x_data is not None:
            return np.array(secondary_x_data)[:, 1:].tolist(), seondary_x_label
        return secondary_x_data, seondary_x_label

    def _load_data(self, cases):
        cols_exclude = cases[0].results.CFL.x_columns
        x_label = path_variable_name(self._x)
        y_label = "residual values"

        x_data = []
        y_data = []

        for case in cases:
            y_variables = [
                f"CFL/{res}"
                for res in case.results.CFL.as_dict().keys()
                if res not in cols_exclude
            ]
            for y in y_variables:
                x_data.append(data_from_path(case, self._x, cases)[1:])
                y_data.append(data_from_path(case, y, cases)[1:])

        return x_data, y_data, x_label, y_label
