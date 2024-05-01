#!/bin/bash

venv_dir="venv"

create_venv() {
    if [ ! -d "$venv_dir" ]; then
        python3 -m venv "$venv_dir"
        echo "venv created"
    fi
}

deactivate_venv() {
    if [ ! -z "$VIRTUAL_ENV" ]; then
        deactivate
        echo "venv deactivated"
    fi
}

activate_venv() {
    deactivate_venv

    if [ -d "$venv_dir" ]; then
        source "$venv_dir/bin/activate"
        echo "venv activated"
    else
        echo "venv not found"
        return 1
    fi
}

install_dependencies() {
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt -qq
    else
        echo "requirements not found"
        return 1
    fi
}

run_pipeline() {
    python data_creation.py &&
    python model_preprocessing.py &&
    python model_preparation.py &&
    python model_testing.py
}

create_venv && activate_venv && install_dependencies && run_pipeline && deactivate_venv || deactivate_venv
