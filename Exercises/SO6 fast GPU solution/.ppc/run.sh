#!/bin/bash
set -e
cat > /box/so.cu
chmod a-w /box/so.cu

# Fix nvprof
cat > "/etc/nsswitch.conf" <<EOT
passwd:         files systemd
EOT

cd /program
/program/.ppc/grader.py --file "/box/so.cu" --binary "/box/so" --json "$@"
