#!/bin/bash
python3 baseline.py -b 32 -j 4\
                    -d market1501\
                    -a resnet50\
                    --evaluate\
                    --resume $1
