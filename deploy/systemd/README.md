# Deployment (systemd user units)

User units, not system units: nothing here needs root, and a user unit dies with your
session rather than fighting your desktop for the CPU at login.

## Install

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service deploy/systemd/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now asa-research.timer asa-run.timer asa-analytics.timer asa-backup.timer
systemctl --user enable --now asa-dashboard.service
```

Then, so the timers keep running when you are not logged in:

```bash
sudo loginctl enable-linger "$USER"
```

## Watch

```bash
systemctl --user list-timers 'asa-*'
journalctl --user -u asa-run.service -f
```

## Notes

* `Nice=10` and `IOSchedulingClass=idle` are deliberate. Rendering saturates all cores for
  ~18 minutes an episode; without these the machine is unusable while it runs.
* `asa-run` is `Type=oneshot` and advances **one** job per firing. A long render simply
  overruns into the next slot, and systemd will not start a second copy of a oneshot unit
  that is still running. The runner's lease is the second line of defence.
* `asa-analytics` is daily, not hourly: YouTube Analytics lags by up to 48 hours and
  polling it more often spends quota to re-read the same numbers.
