# TimeTracker — Notes for Claude

## Launch / Admin Elevation

`TimeTracker.bat` intentionally re-launches itself as Administrator. **Do not remove the UAC elevation.**

The app binds to `0.0.0.0:5001` so it is reachable by other machines on the local network. Windows requires elevated privileges to allow inbound connections through the firewall on that port.
