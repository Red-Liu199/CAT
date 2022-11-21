from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import platform

tools_dir = './src/'

system_name = platform.system()  # find the OS type of the running platform
if system_name == "Windows":  # for Windows
    print("Call Windows tasks")
    ext_modules = [
        Extension('trie',  # name of extension module
                  sources=['trie.pyx'],  # source code (content) of the extension module
                  include_dirs=[tools_dir],  # put the compiled code into which document
                  language='c++',  # language of the compiled code
                  extra_compile_args=['/openmp'],
                  extra_link_args=['/openmp'])
    ]
else:  # for others
    ext_modules = [
        Extension('trie',
                  sources=['trie.pyx'],
                  include_dirs=[tools_dir],
                  language='c++',
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp'])
    ]

setup(
    name='trie',  # set up the model based on ext_modules
    ext_modules=cythonize(ext_modules, language_level = "3")
)
