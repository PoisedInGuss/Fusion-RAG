#!/bin/bash
# Build script for Java QPP
# Real QPP implementations (NQC, WIG, SMV, SigmaMax, etc.)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building QPP Java code..."

# Build with Maven
mvn compile dependency:copy-dependencies -DoutputDirectory=target/dependency -q

echo ""
echo "Testing QPP Bridge..."
echo '{"query":"test query","documents":[{"score":0.9},{"score":0.7},{"score":0.5}],"methods":["nqc","smv","SigmaMax"]}' | \
    java -cp "target/classes:target/dependency/*" qpp.QPPBridge

echo ""
echo "✅ Build successful! QPP is ready."
echo "   Classes: $SCRIPT_DIR/target/classes"
echo "   Dependencies: $SCRIPT_DIR/target/dependency"
