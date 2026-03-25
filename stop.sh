#!/usr/bin/env bash
set -e

# Stop ArduPilot SITL + Webots simulation containers

echo "🛑 Stopping simulation containers..."

if command -v podman-compose &> /dev/null; then
    podman-compose down
elif command -v docker &> /dev/null; then
    docker compose down
else
    echo "❌ Error: Neither podman-compose nor docker found."
    exit 1
fi

echo "✅ Containers stopped."
