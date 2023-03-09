from setuptools import setup

package_name = 'yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mostafa',
    maintainer_email='moustafa.aldilati.1999@gmail.com',
    description='This package gets rectified images from the zed camera and passes it through a YOLO model',

    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo = yolo.yolo_node:main'
        ],
    },
)
