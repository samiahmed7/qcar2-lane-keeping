#!/usr/bin/env bash
# Close any rqt windows opened by view_in_rqt.sh.
pkill -9 -f 'rqt_image_view' 2>/dev/null
pkill -9 -f 'rqt_plot'       2>/dev/null
pkill -9 -f 'rqt_graph'      2>/dev/null
sleep 1
echo "rqt windows closed"
