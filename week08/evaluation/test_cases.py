from dataclasses import dataclass


@dataclass
class TestCase:
    name: str
    goal: str
    expected_agents: list[str]
    required_concepts: list[str]


TEST_CASES = [
    TestCase(
        name="Visual Odometry",
        goal=(
            "Explain the advantages and limitations of visual odometry "
            "for GPS-denied drone navigation."
        ),
        expected_agents=[
            "research",
            "analyst",
            "writer",
        ],
        required_concepts=[
            "visual odometry",
            "gps-denied",
            "drift",
            "lighting",
            "imu",
        ],
    ),
]


# Keep only one test while validating.
# Restore the other two after validation.

# TEST_CASES = TEST_CASES[:1]