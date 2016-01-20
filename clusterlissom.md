# Introduction #

`clusterlissom` is a project to enable stream simulation of a hierarchical LISSOM model (many LISSOM maps arranged in an hierarchical structure) on a cluster of GPU enabled computers.
An interesting feature of the system is that it lets the developer describe the hierarchy and the maps' properties from an human readable configuration file, which really helps designing good experiments (also using the software on a single computer, without taking advantage of its streaming capabilities).



NOTE: in the source code available for download, the really important thing to change while testing is within the `example` directory. Also remember to check `run.sh` for usage.

Also remember to copy a fresh liblissom.a and lissom.h file within the directory, to make sure it will work (no problem for compiling under Linux, but better doing it).