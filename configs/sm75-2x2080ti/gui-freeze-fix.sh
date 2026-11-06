#!/usr/bin/env bash
# Take the NVIDIA cards out of the desktop's way (root, one shot, needs a reboot).
#
# Why: the display is on the AMD FirePro W4100 (radeon, /dev/dri/card0). The two RTX 2080 Ti
# register as DRM devices too (card1, card2 - nvidia-drm), so gnome-shell hotplugs them and the
# NVIDIA driver's modeset takes the global DRM/atomic locks. While XTier's compute runs, the
# driver does KMS work on a busy card, the AMD card's KMS waits for the lock, and the desktop
# freezes (the image stops; the shell then dies with "Failed to hotplug secondary gpu
# .../renderD129: ENODEV", "g_hash_table_insert_internal: assertion 'hash_table != NULL' failed"
# and "Failed to make thread 'KMS thread' normally scheduled: Device or resource busy").
# The cards drive no output ("nvidia 0000:03:00.0: [drm] Cannot find any crtc or sizes"), so
# removing them from the DRM path costs nothing: CUDA and nvidia-smi do not need nvidia-drm.
#
# Run:  sudo bash configs/sm75-2x2080ti/gui-freeze-fix.sh   then reboot.
set -euo pipefail

echo "== 1. the NVIDIA driver keeps out of the DRM/KMS path =="
cat >/etc/modprobe.d/blacklist-nvidia-drm.conf <<'EOF'
# The desktop runs on the AMD FirePro W4100 (radeon). The NVIDIA cards are compute-only:
# without nvidia-drm they never appear as /dev/dri/card*, so the driver takes no DRM lock and
# gnome-shell never hotplugs them (that hotplug is what wedges the compositor).
blacklist nvidia_drm
blacklist nvidia_modeset
EOF
# Belt and braces: if a later update/initramfs loads the module anyway, keep it out of modesetting.
if ! grep -q 'nvidia-drm.modeset=0' /etc/default/grub; then
  sed -i 's/^\(GRUB_CMDLINE_LINUX_DEFAULT=".*\)"$/\1 nvidia-drm.modeset=0"/' /etc/default/grub
fi
update-initramfs -u
update-grub
echo "   /etc/modprobe.d/blacklist-nvidia-drm.conf written, initramfs and grub updated"

echo "== 2. DCGM's host engine stops polling the GPUs =="
# nv-hostengine polls both cards through NVML every few seconds (the endless apparmor denials in
# the journal). It is not needed here and it touches the GPUs while XTier computes on them.
snap stop dcgm.nv-hostengine || true
snap disable dcgm.nv-hostengine || true

echo "== 3. the driver stays loaded (no hotplug churn) =="
nvidia-smi -pm 1 || true

cat <<'EOF'

== reboot now, then check ==

  ls /dev/dri/                     # only card0 (AMD) and renderD128: the NVIDIA cards are gone
  journalctl -b 0 | grep -iE 'hotplug secondary|KMS thread|hash_table'   # nothing: no hotplug
  nvidia-smi                       # both cards still there for CUDA

The watchdog (configs/sm75-2x2080ti/gui-watch.sh, xtier-gui-watch.service) restarts a wedged
gnome-shell on its own; it is a safety net, not the fix.
EOF
