from setuptools import setup, find_packages

setup(
    name='mr_toolkit',  # 项目名称
    version='0.1.0',  # 项目版本
    description='A short description of your project',  # 项目简短描述
    author='Mingyang Song',  # 作者名
    author_email='mysong23@m.fudan.edu.cn',  # 作者邮箱
    url='https://github.com/yourusername/your_project',  # 项目主页 URL
    packages=find_packages(),  # 自动发现并包含项目中的所有包
    install_requires=[  # 项目的依赖包列表
        'numpy',  
        'requests',  
    ],
    classifiers=[  # 项目分类标签（提高可见度）
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',  # 选择合适的许可证
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  # 项目支持的 Python 版本
    include_package_data=True,  # 包含非 Python 文件
    long_description=open('README.md').read(),  # 从 README 文件中读取长描述
    long_description_content_type='text/markdown',  # 长描述的格式
)