#!/usr/bin/env bash
# Watchdog for a wedged GNOME shell (the safety net under configs/sm75-2x2080ti/gui-freeze-fix.sh).
#
# The freeze: gnome-shell's compositor stops answering, so the screen stops updating. On X11 the
# session manager (gnome-session) restarts gnome-shell when it dies, so killing it brings the
# desktop back with the apps still running. On Wayland that would end the session, so there the
# watchdog only logs.
#
# The health check is the shell's own D-Bus method: a wedged shell's main loop does not answer.
set -u
LOG=${XDG_STATE_HOME:-$HOME/.local/state}/xtier-gui-watch.log
BUS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/bus}
INTERVAL=${XTIER_GUI_WATCH_INTERVAL:-20}
TIMEOUT=${XTIER_GUI_WATCH_TIMEOUT:-12}
NEEDED=${XTIER_GUI_WATCH_FAILS:-3}     # consecutive misses before the shell is restarted
mkdir -p "$(dirname "$LOG")"

say() { echo "$(date -Is) $*" >>"$LOG"; }

alive() {
  timeout "$TIMEOUT" gdbus call --address "$BUS" --dest org.gnome.Shell \
    --object-path /org/gnome/Shell --method org.gnome.Shell.Eval "1+1" >/dev/null 2>&1
}

miss=0
while :; do
  sleep "$INTERVAL"
  pgrep -x gnome-shell >/dev/null || continue
  if alive; then
    [ "$miss" -gt 0 ] && say "the shell answers again"
    miss=0
  else
    miss=$((miss + 1))
    say "the shell did not answer ($miss/$NEEDED)"
    if [ "$miss" -ge "$NEEDED" ]; then
      if [ -z "${WAYLAND_DISPLAY:-}" ]; then
        say "restarting gnome-shell (X11: gnome-session starts it again)"
        pkill -x gnome-shell || true
      else
        say "gnome-shell is wedged on Wayland: restart the session by hand"
      fi
      miss=0
    fi
  fi
done
