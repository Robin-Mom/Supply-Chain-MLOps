#!/bin/bash
# freeze.sh
EXCLUDE="torch|torchvision|torchaudio|triton|nvidia"

pip freeze | grep -Ev "$EXCLUDE" > requirements.txt
echo "✅ requirements.txt généré sans les packages GPU/torch"