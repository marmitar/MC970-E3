# CUDA Labs

This repository contains a set of three labs focused on CUDA parallel programming, split into separate folders, each containing a specific lab exercise.

## Lab Exercise Structure

Each lab is located in a separate folder and contains the necessary files and instructions to complete it. The instructions for each lab exercise are located in a README file in the respective exercise folder.

The labs are designed to be completed in order, as each lab builds upon the previous one. Therefore, it is recommended that you start with the first lab and work your way through them in sequence.

## Running on CENAPAD cluster

If you wish, you can run the lab exercises on the CENAPAD cluster, a high-performance computing cluster available for academic purposes. The cluster has GPUs and CUDA libraries pre-installed, which can help you speed up your GPU assignments.

## Getting Started

To get started, clone this repository to your local machine using the following command:

```sh
git clone --recurse-submodules <repository URL>
```

The command `--recurse-submodules` is used to clone the test folder, which is from a separate repository, because of its size.

If you have already cloned the repository without the --recurse-submodules option, you can still clone the submodules by running the following command in the repository's root directory:

```sh
git submodule update --init --recursive
```

Once the repository is cloned, navigate to the first lab folder and read the instructions in the README file. Follow the instructions to complete the lab, and then move on to the next lab in the sequence.

## Prerequisites

To complete these labs, you will need a basic understanding of C or C++ programming language and the fundamentals of parallel programming concepts. You will also need a development environment for the language used in the labs and CUDA libraries installed.

## Contributing

If you find any issues with the labs or have suggestions for improvements, feel free to create an issue or pull request on this repository.
