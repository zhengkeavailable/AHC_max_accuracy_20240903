import setuptools

packages = ['ahc_packages']  # 唯一的包名，自己取名
setuptools.setup(name='ahc_packages',
                 version='1.0',
                 author='zhengke',
                 packages=packages,
                 package_dir={'requests': 'requests'}, )
