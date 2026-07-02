from setuptools import find_packages, setup

package_name = 'hazmat_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=["hazmat_detection", "hazmat_detection/HazmatAnalyzer", "hazmat_detection/HazmatAnalyzer/src/detection", "hazmat_detection/HazmatAnalyzer/src/classification", "hazmat_detection/HazmatAnalyzer/configs", "hazmat_detection/HazmatAnalyzer/models"],
    package_data={
        "hazmat_detection/HazmatAnalyzer/models": ["*.pt", "*.onnx"]
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='ros@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'run_hazmat = hazmat_detection.detect:main',
            'run_better_hazmat = hazmat_detection.better_detect:main'
        ],
    },
)
