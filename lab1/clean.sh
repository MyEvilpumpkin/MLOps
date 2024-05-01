#!/bin/bash

clean_data() {
    rm -r train test
}

clean_models() {
    rm -r models
}

clean_venv() {
    rm -r venv
}

while getopts dmv flag
do
    case "${flag}" in
        d) clean_data;;
        m) clean_models;;
        v) clean_venv;;
    esac
done
