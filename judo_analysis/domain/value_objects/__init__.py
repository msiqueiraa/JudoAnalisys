"""Value objects de dominio."""

from .comparison import ComparisonResult, MetricComparison
from .gi_color import GiColor
from .joint_angles import JointAngles
from .speed import Speed

__all__ = [
    "ComparisonResult",
    "GiColor",
    "JointAngles",
    "MetricComparison",
    "Speed",
]
