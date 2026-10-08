import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():
    package_share = get_package_share_directory(
        'so101_description'
    )

    urdf_file = os.path.join(
        package_share,
        'urdf',
        'so101.urdf'
    )

    rviz_config_file = os.path.join(
        package_share,
        'rviz',
        'so101_demo.rviz'
    )

    with open(urdf_file, 'r', encoding='utf-8') as file:
        robot_description = file.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description':ParameterValue(robot_description, value_type=str)
        }]
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])
