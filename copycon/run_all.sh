#!/bin/bash

python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops regular  & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops regular --oles True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops regular --fairdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops regular --sdarts random & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops regular --pcdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops regular  & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops regular --oles True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops regular --fairdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops regular --sdarts random & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops regular --pcdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops regular  & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops regular --oles True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops regular --fairdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops regular --sdarts random & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops regular --pcdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops all_skip  & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops all_skip --oles True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops all_skip --fairdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops all_skip --sdarts random & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops all_skip --pcdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops all_skip  & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops all_skip --oles True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops all_skip --fairdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops all_skip --sdarts random & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops all_skip --pcdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops all_skip  & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops all_skip --oles True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops all_skip --fairdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops all_skip --sdarts random & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops all_skip --pcdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops no_skip  & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops no_skip --oles True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops no_skip --fairdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops no_skip --sdarts random & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace wide --ops no_skip --pcdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops no_skip  & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops no_skip --oles True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops no_skip --fairdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops no_skip --sdarts random & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace deep --ops no_skip --pcdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops no_skip  & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops no_skip --oles True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops no_skip --fairdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops no_skip --sdarts random & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer darts --subspace single_cell --ops no_skip --pcdarts True & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace wide --ops regular & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace deep --ops regular & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace single_cell --ops regular & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace wide --ops all_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace deep --ops all_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace single_cell --ops all_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace wide --ops no_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace deep --ops no_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer drnas --subspace single_cell --ops no_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace wide --ops regular & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace deep --ops regular & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace single_cell --ops regular & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace wide --ops all_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace deep --ops all_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace single_cell --ops all_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace wide --ops no_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace deep --ops no_skip & sleep 5
python launch_supernet_search.py --seeds 0,1,2 --dataset cifar10_supernet --tag first-full-run --optimizer gdas --subspace single_cell --ops no_skip & sleep 5

