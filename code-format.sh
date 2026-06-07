#!/bin/bash

CODE_STYLE="{BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 120, Standard: Cpp11}"

find include/core src -name *.h -or -name *.cc | while read line
do
    clang-format -style="${CODE_STYLE}" -i ${line}
done
