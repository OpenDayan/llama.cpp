#!/bin/bash

./build/bin/llama-batched-bench -m ../DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-E4M3.gguf \
                                -t 1 \
                                -c 2048 \
                                -b 3 \
                                -npp 6 \
                                -ntg 4 \
                                -ngl 28 \
                                -npl 1 \
                                -ctk bf16 \
                                -ctv bf16