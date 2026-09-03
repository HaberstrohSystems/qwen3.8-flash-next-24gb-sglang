#!/bin/bash
# TCP forwarder so DeepSWE containers (docker bridge) reach the local SGLang server bound to 127.0.0.1:30000.
exec socat TCP-LISTEN:30001,bind=172.17.0.1,fork,reuseaddr TCP:127.0.0.1:30000
