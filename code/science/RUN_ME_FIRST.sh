#!/bin/bash

cat split_feature_chunks/x* > features.tar.gz
tar xzf features.tar.gz
rm features.tar.gz
