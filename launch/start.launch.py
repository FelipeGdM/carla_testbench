from launch_ros.substitutions import FindPackageShare

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    PathJoinSubstitution,
    TextSubstitution,
    LaunchConfiguration,
)


def generate_launch_description():

    carla_spawn_objects_args = [
        DeclareLaunchArgument(
            name="objects_definition_file",
            default_value=os.path.join(
                get_package_share_directory("carla_testbench"),
                "config",
                "objects.json",
            ),
        ),
        DeclareLaunchArgument(name="spawn_point_ego_vehicle", default_value=""),
        DeclareLaunchArgument(name="spawn_sensors_only", default_value="False"),
    ]

    carla_spawn_objects_subs = {
        arg.name: LaunchConfiguration(arg.name)
        for arg in carla_spawn_objects_args
    }

    carla_spawn_objects = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("carla_spawn_objects"),
                        "carla_spawn_objects.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments=carla_spawn_objects_subs.items(),
    )

    return LaunchDescription([*carla_spawn_objects_args, carla_spawn_objects])