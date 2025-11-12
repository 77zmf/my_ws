from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Whether to use simulation time'
    )
    
    # Get rviz config path
    default_rviz_config_path = PathJoinSubstitution([
        FindPackageShare('manual_calib'),
        'config',
        'rviz_config.rviz'
    ])
    
    # Get config.yaml path
    default_config_path = PathJoinSubstitution([
        FindPackageShare('manual_calib'),
        'config',
        'config.yaml'
    ])
    
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_config_path,
        description='Path to RViz2 config file'
    )
    
    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_path,
        description='Path to config.yaml file'
    )
    
    # Node parameters
    node_params = {
        'rate': 10.0,
    }
    
    # Create the manual calibration node
    manual_calib_node = Node(
        package='manual_calib',
        executable='manual_calib_node',
        name='manual_calib_node',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'config_file': LaunchConfiguration('config_file'),
            **node_params
        }],
        output='screen'
    )
    
    # Start rviz2
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }],
        output='screen'
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        rviz_config_arg,
        config_arg,
        manual_calib_node,
        rviz2_node,
    ])

