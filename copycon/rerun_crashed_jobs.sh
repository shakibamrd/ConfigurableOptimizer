#!/bin/bash

python launch_supernet_search.py --seeds 0 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops regular --oles True & sleep 5
python launch_supernet_search.py --seeds 0 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace single_cell --ops no_skip & sleep 5
python launch_supernet_search.py --seeds 0 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace wide --ops regular & sleep 5
python launch_supernet_search.py --seeds 0 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace deep --ops regular & sleep 5
python launch_supernet_search.py --seeds 0 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace single_cell --ops regular & sleep 5
python launch_supernet_search.py --seeds 0 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace wide --ops all_skip & sleep 5
python launch_supernet_search.py --seeds 0 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace deep --ops all_skip & sleep 5
python launch_supernet_search.py --seeds 0 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace deep --ops no_skip & sleep 5
python launch_supernet_search.py --seeds 0 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace wide --ops regular & sleep 5
python launch_supernet_search.py --seeds 1 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops no_skip --fairdarts True & sleep 5
python launch_supernet_search.py --seeds 1 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops regular  & sleep 5
python launch_supernet_search.py --seeds 2 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace single_cell --ops all_skip & sleep 5

