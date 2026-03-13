#!/bin/bash

./build/bin/llama-cli -m ../DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-E4M3.gguf \
                      -p "Hello" \
                      -s 1 \
                      -t 1 \
                      -n 5 \
                      -ngl 28 \
                      -ctk bf16 \
                      -ctv bf16 \
                      -st \
                      --no-warmup