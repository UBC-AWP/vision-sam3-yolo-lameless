#!/usr/bin/env python3
"""Test tenant name extraction logic"""
key = "Farm1/raw/MVI_0115_short_0.MP4"
parts = key.split("/")
print(f"Key: {key}")
print(f"Parts: {parts}")
print(f"parts[0]: {parts[0]}")
print(f"parts[-2]: {parts[-2]}")
print(f"parts[-1]: {parts[-1]}")
print(f"len(parts): {len(parts)}")
print(f"Is direct upload: {len(parts) >= 3 and parts[-2] == 'raw' and parts[-1].endswith(('.mp4', '.MP4')) and parts[-1] != 'original.mp4'}")

# Simulate tenant_name_to_id
tenant_name_to_id = {
    "Farm1": "da656f12-c569-4272-8183-fea6225e105f",
    "Farm2": "0feb578c-76c1-4604-9ae3-ccb8f1626f68",
    "default": "b0000000-0000-0000-0000-000000000001"
}

potential_tenant_name = parts[0]
print(f"\npotential_tenant_name: {potential_tenant_name}")
print(f"In tenant_name_to_id: {potential_tenant_name in tenant_name_to_id}")
if potential_tenant_name in tenant_name_to_id:
    print(f"tenant_id: {tenant_name_to_id[potential_tenant_name]}")
