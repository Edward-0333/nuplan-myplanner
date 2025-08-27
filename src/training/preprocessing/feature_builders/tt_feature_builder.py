import warnings
from typing import List, Type

import numpy as np
import shapely
import torch

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from shapely import LineString, Point

from src.training.preprocessing.features.tt_feature import TTFeature


class TTFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        radius: float = 100,

    ) -> None:
        super().__init__()

        self.radius = radius

    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return TTFeature  # type: ignore

    def get_class(self) -> Type[AbstractFeatureBuilder]:
        return TTFeatureBuilder

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "feature"

    def get_features_from_scenario(
        self,
        scenario: AbstractScenario,
        iteration=0,
    ) -> AbstractModelFeature:

        return 1

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> AbstractModelFeature:

        return 1

    def _build_feature(
        self,
    ):

        return 1
