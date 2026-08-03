#!/usr/bin/env python3
"""Patch a Container App or Container App Job to add an AzureFile volume mount.

Reads the resource JSON from stdin, adds/overwrites volumes + volumeMounts,
and writes the resulting YAML to /tmp/job-patch.yaml.
"""
import sys, json, yaml

d = json.load(sys.stdin)
t = d.get("properties", {}).get("template", {})
t["volumes"] = [{"name": "runs-volume", "storageName": "runs-volume", "storageType": "AzureFile"}]
t["containers"][0]["volumeMounts"] = [{"volumeName": "runs-volume", "mountPath": "/mnt/runs"}]
d.pop("systemData", None)

with open("/tmp/job-patch.yaml", "w") as f:
    yaml.dump(d, f, default_flow_style=False)

print("ok")
