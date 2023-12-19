class RampCFL(Flow360BaseModel):
    """
    Ramp CFL for time stepping component
    """

    type: str = pd.Field("ramp", const=True)
    initial: Optional[PositiveFloat] = pd.Field()
    final: Optional[PositiveFloat] = pd.Field()
    ramp_steps: Optional[int] = pd.Field(alias="rampSteps")
    randomizer: Optional[Dict] = pd.Field()

    @classmethod
    def default_steady(cls):
        """
        returns default steady CFL settings
        """
        return cls(initial=5, final=200, ramp_steps=40)

    @classmethod
    def default_unsteady(cls):
        """
        returns default unsteady CFL settings
        """
        return cls(initial=1, final=1e6, ramp_steps=30)


class AdaptiveCFL(Flow360BaseModel):
    """
    Adaptive CFL for time stepping component
    """

    type: str = pd.Field("adaptive", const=True)
    min: Optional[PositiveFloat] = pd.Field(default=0.1)
    max: Optional[PositiveFloat] = pd.Field(default=10000)
    max_relative_change: Optional[PositiveFloat] = pd.Field(alias="maxRelativeChange", default=1)
    convergence_limiting_factor: Optional[PositiveFloat] = pd.Field(
        alias="convergenceLimitingFactor", default=0.25
    )


# pylint: disable=E0213
class BaseTimeStepping(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for time stepping component
    """

    max_pseudo_steps: Optional[pd.conint(gt=0, le=100000)] = pd.Field(2000, alias="maxPseudoSteps")
    CFL: Optional[Union[RampCFL, AdaptiveCFL]] = pd.Field()

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> BaseTimeStepping:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [DeprecatedAlias(name="physical_steps", deprecated="maxPhysicalSteps")]


class SteadyTimeStepping(BaseTimeStepping):
    """
    Steady time stepping component
    """

    physical_steps: Literal[1] = pd.Field(1, alias="physicalSteps", const=True)
    time_step_size: Literal["inf"] = pd.Field("inf", alias="timeStepSize", const=True)


class UnsteadyTimeStepping(BaseTimeStepping):
    """
    Unsteady time stepping component
    """

    physical_steps: Optional[PositiveInt] = pd.Field(alias="physicalSteps")
    time_step_size: Optional[TimeType.Positive] = pd.Field(alias="timeStepSize")


TimeStepping = Union[SteadyTimeStepping, UnsteadyTimeStepping]
