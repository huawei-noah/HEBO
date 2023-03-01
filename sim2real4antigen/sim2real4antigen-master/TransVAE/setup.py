import os
from setuptools import setup, find_packages

PACKAGES = find_packages()

ver_file = os.path.join('transvae', 'version.py')
with open(ver_file) as f:
    exec(f.read())

with open('README.md') as readme_file:
    README = readme_file.read()

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=README,
            long_description_content_type=CONTENT_TYPE,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            install_requires=REQUIRES
            )


if __name__ == '__main__':
    setup(**opts)
