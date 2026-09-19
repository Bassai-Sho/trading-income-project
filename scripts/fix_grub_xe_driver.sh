#!/usr/bin/env bash
# ============================================================================
# Fix: Migrate Intel Arc iGPU from i915 to xe kernel driver
# Root cause: i915 10s fence watchdog kills long GPU prefill (27B model, ~25-40s)
# xe driver supports native preemption and does not have this watchdog limit.
#
# Reference: OpenVINO Issue #36260, #36404
# System: adam-NUC14RVH-B, Intel Core Ultra Meteor Lake, Arc 140V iGPU
# ============================================================================
set -e

echo "=== Step 1: Confirm GPU PCI ID ==="
lspci -nn -s 00:02.0
GPU_ID=$(lspci -nn -s 00:02.0 | grep -oP '\[8086:\K[0-9a-f]+(?=\])')
echo "GPU device ID: $GPU_ID"

echo ""
echo "=== Step 2: Check current driver ==="
lspci -k -s 00:02.0 | grep "Kernel driver"

echo ""
echo "=== Step 3: Update GRUB ==="
GRUB_FILE="/etc/default/grub"
CURRENT=$(grep GRUB_CMDLINE_LINUX_DEFAULT "$GRUB_FILE")
echo "Current: $CURRENT"

# Check if already applied
if grep -q "xe.force_probe" "$GRUB_FILE"; then
    echo "✅ xe driver already configured in GRUB"
else
    sudo sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\"\(.*\)\"/GRUB_CMDLINE_LINUX_DEFAULT=\"\1 i915.enable_psr=0 i915.force_probe=!${GPU_ID} xe.force_probe=${GPU_ID}\"/" "$GRUB_FILE"
    echo "✅ GRUB updated"
    sudo update-grub
    echo "✅ GRUB configuration regenerated"
fi

echo ""
echo "=== Step 4: Verify GRUB change ==="
grep GRUB_CMDLINE_LINUX_DEFAULT "$GRUB_FILE"

echo ""
echo "============================================================"
echo "REBOOT REQUIRED: sudo reboot"
echo ""
echo "After reboot, verify with:"
echo "  lspci -k -s 00:02.0    (must show: Kernel driver in use: xe)"
echo "  python3 -c \"import openvino as ov; print(ov.Core().available_devices)\""
echo ""
echo "Rollback if needed:"
echo "  sudo sed -i 's/ i915.enable_psr=0 i915.force_probe=!${GPU_ID} xe.force_probe=${GPU_ID}//' /etc/default/grub"
echo "  sudo update-grub && sudo reboot"
echo "============================================================"
